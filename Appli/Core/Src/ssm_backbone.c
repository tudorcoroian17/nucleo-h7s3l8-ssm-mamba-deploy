#include "ssm_backbone.h"

#include <math.h>
#include <string.h>

static inline float ssm_softplus(float x) {
	/* 	Matches torch.nn.functional.softplus's default threshold=20: linear
	 * 	for large x, where log1p(exp(x)) would lose precision anyway. */
	return (x > 20.0f) ? x : log1pf(expf(x));
}

static inline float ssm_silu(float x) {
	return x / (1.0f + expf(-x));
}

static inline float ssm_quant_dequant(float v, float scale) {
	float q = roundf(v / scale);
	if (q > 127.0f) q = 127.0f;
	if (q < -127.0f) q = -127.0f;
	return q * scale;
}

static void ssm_rmsnorm(const float *x, const ssm_norm_weights_t *nw, float *out) {
	float ss = 0.0f;
	for (int i = 0; i < SSM_D_MODEL; i++) {
		ss += x[i] * x[i];
	}
	float inv_rms = 1.0f / sqrtf(ss / (float) SSM_D_MODEL + 1e-5f);
	for (int i = 0; i < SSM_D_MODEL; i++) {
		out[i] = x[i] * inv_rms * SSM_NORM_W(nw, i);
	}
}

static void ssm_block_step(
		const ssm_block_weights_t *w,
		float h[SSM_D_INNER][SSM_D_STATE],
		float conv_hist[SSM_D_INNER][SSM_D_CONV - 1],
		const float *x_norm,
		float *block_out) {

	float u_raw[SSM_D_INNER];
	float z[SSM_D_INNER];
	float u[SSM_D_INNER];
	float x_dbl[SSM_DT_RANK + 2 * SSM_D_STATE];
	float delta_low[SSM_DT_RANK];
	float B[SSM_D_STATE];
	float C[SSM_D_STATE];
	float delta[SSM_D_INNER];
	float y[SSM_D_INNER];

	/* in_proj: split into u_raw (rows 0..D_INNER-1) and z (rows D_INNER..).
	 * SSM_QUANT_Z is applied once z[] is fully computed -- boundaries group,
	 * no-op when this scheme doesn't quantize activations. */
	for (int o = 0; o < SSM_D_INNER; o++) {
		float acc = 0.0f;
		for (int i = 0; i < SSM_D_MODEL; i++) {
			acc += SSM_IN_PROJ_W(w, o, i) * x_norm[i];
		}
		u_raw[o] = acc;
	}
	for (int o = 0; o < SSM_D_INNER; o++) {
		float acc = 0.0f;
		for (int i = 0; i < SSM_D_MODEL; i++) {
			acc += SSM_IN_PROJ_W(w, SSM_D_INNER + o, i) * x_norm[i];
		}
		z[o] = acc;
	}
	for (int o = 0; o < SSM_D_INNER; o++) {
		z[o] = SSM_QUANT_Z(w, z[o]);
	}

	/*	causal depthwise conv: kernel tap 0 = oldest (t-3) .. tap 3 = current
	 * 	(t), then SiLU. conv_hist holds t-3,t-2,t-1 oldest-first.
	 * 	SSM_QUANT_U applied once u[] (post-conv, post-SiLU -- Python's
	 * 	u_post_conv_silu) is fully computed. */
	for (int c = 0; c < SSM_D_INNER; c++) {
		float acc = SSM_CONV_B(w, c);
		for (int k = 0; k < SSM_D_CONV - 1; k++) {
			acc += SSM_CONV_W(w, c, k) * conv_hist[c][k];
		}
		acc += SSM_CONV_W(w, c, SSM_D_CONV - 1) * u_raw[c];
		u[c] = ssm_silu(acc);

		for (int k = 0; k < SSM_D_CONV - 2; k++) {
			conv_hist[c][k] = conv_hist[c][k + 1];
		}
		conv_hist[c][SSM_D_CONV - 2] = u_raw[c];
	}
	for (int c = 0; c < SSM_D_INNER; c++) {
		u[c] = SSM_QUANT_U(w, u[c]);
	}

	/* x_proj on post-conv u -> delta_low / B / C */
	for (int o = 0; o < SSM_DT_RANK + 2 * SSM_D_STATE; o++) {
		float acc = 0.0f;
		for (int i = 0; i < SSM_D_INNER; i++) {
			acc += SSM_X_PROJ_W(w, o, i) * u[i];
		}
		x_dbl[o] = acc;
	}
	memcpy(delta_low, &x_dbl[0], sizeof(delta_low));
	memcpy(B, &x_dbl[SSM_DT_RANK], sizeof(B));
	memcpy(C, &x_dbl[SSM_DT_RANK + SSM_D_STATE], sizeof(C));

	/* dt_proj + softplus */
	for (int o = 0; o < SSM_D_INNER; o++) {
		float acc = SSM_DT_PROJ_B(w, o);
		for (int i = 0; i < SSM_DT_RANK; i++) {
			acc += SSM_DT_PROJ_W(w, o, i) * delta_low[i];
		}
		delta[o] = ssm_softplus(acc);
	}

	/* 	euler discretize + recurrence + output, fused per timestep -- matches
	 * 	ssm_block.py's discretize(discretization="euler") +
	 * 	_scan_streaming(): A_bar = 1 + clamp(delta*A, min=-1.9),
	 * 	B_bar = delta*B. SSM_A dequantizes AND applies -exp(A_log) inline
	 * 	when that field is quantized -- see ssm_weights.h.
	 *
	 * 	y[c] here is Python's y_gated (post D-shortcut, post SiLU gate) --
	 * 	SSM_QUANT_Y_GATED applies right after it's computed. y_c (the raw
	 * 	scan output, pre-shortcut-and-gate -- Python's y_scan) has no
	 * 	quantize hook: it's a scalar consumed on the same line it's
	 * 	produced, not a materialized array. That's scan-group work, not
	 * 	boundaries -- deliberately not wired yet. */
	for (int c = 0; c < SSM_D_INNER; c++) {
		float y_c = 0.0f;
		for (int n = 0; n < SSM_D_STATE; n++) {
			float deltaA = delta[c] * SSM_A(w, c, n);
			if (deltaA < -1.9f) {
				deltaA = -1.9f;
			}
			float A_bar = 1.0f + deltaA;
			float B_bar = delta[c] * B[n];
			float new_h = A_bar * h[c][n] + B_bar * u[c];
			h[c][n] = new_h;
			y_c += new_h * C[n];
		}
		y[c] = (y_c + u[c] * SSM_D_PARAM(w, c)) * ssm_silu(z[c]);
		y[c] = SSM_QUANT_Y_GATED(w, y[c]);
	}

	for (int o = 0; o < SSM_D_MODEL; o++) {
		float acc = 0.0f;
		for (int i = 0; i < SSM_D_INNER; i++) {
			acc += SSM_OUT_PROJ_W(w, o, i) * y[i];
		}
		block_out[o] = acc;
	}
	for (int o = 0; o < SSM_D_MODEL; o++) {
		block_out[o] = SSM_QUANT_BLOCK_OUT(w, block_out[o]);
	}
}

void SSMBackbone_Reset(SSMBackbone_State *state) {
	memset(state, 0, sizeof(*state));
}

void SSMBackbone_ProcessFrame(SSMBackbone_State *state, const float *frame_in, float *final_norm_out) {
	float x[SSM_D_MODEL];
	float x_norm[SSM_D_MODEL];
	float block_out[SSM_D_MODEL];

	memcpy(x, frame_in, sizeof(x));

	for (int layer = 0; layer < SSM_N_LAYERS; layer++) {
		ssm_rmsnorm(x, &ssm_blocks[layer].norm_w, x_norm);
		ssm_block_step(&ssm_blocks[layer], state->h[layer], state->conv_hist[layer], x_norm, block_out);
		for (int i = 0; i < SSM_D_MODEL; i++) {
			x[i] += block_out[i];
		}
	}

	float normed[SSM_D_MODEL];
	ssm_rmsnorm(x, &ssm_final_norm_w, normed);
	for (int i = 0; i < SSM_D_MODEL; i++) {
		normed[i] = SSM_QUANT_FINAL_NORM(normed[i]);
	}

	for (int i = 0; i < SSM_D_MODEL; i++) {
		state -> pooled_sum[i] += normed[i];
	}
	state->frame_count++;

	if (final_norm_out != NULL) {
		memcpy(final_norm_out, normed, sizeof(normed));
	}
}

void SSMBackbone_GetPooled(const SSMBackbone_State *state, float *pooled_out) {
	float inv_n = 1.0f / (float) state->frame_count;
	for (int i = 0; i < SSM_D_MODEL; i++) {
		pooled_out[i] = state->pooled_sum[i] * inv_n;
	}
}

void SSMBackbone_NormalizeFrame(const float *raw, float *normalized) {
    for (int i = 0; i < SSM_D_MODEL; i++) {
        normalized[i] = (raw[i] - ssm_norm_mean[i]) / ssm_norm_std[i];
    }
}