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

static void ssm_rmsnorm(const float *x, const float *weigth, float *out) {
	float ss = 0.0f;
	for (int i = 0; i < SSM_D_MODEL; i++) {
		ss += x[i] * x[i];
	}
	float inv_rms = 1.0f / sqrtf(ss / (float) SSM_D_MODEL + 1e-5f);
	for (int i = 0; i < SSM_D_MODEL; i++) {
		out[i] = x[i] * inv_rms * weigth[i];
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

	/* in_proj: split into u_raw (rows 0..D_INNER-1) and z (rows D_INNER..) */
	for (int o = 0; o < SSM_D_INNER; o++) {
		float acc = 0.0f;
		const float *row = &w->in_proj_w[o * SSM_D_MODEL];
		for (int i = 0; i < SSM_D_MODEL; i++) {
			acc += row[i] * x_norm[i];
		}
		u_raw[o] = acc;
	}
	for (int o = 0; o < SSM_D_INNER; o++) {
		float acc = 0.0f;
		const float *row = &w->in_proj_w[(SSM_D_INNER + o) * SSM_D_MODEL];
		for (int i = 0; i < SSM_D_MODEL; i++) {
			acc += row[i] * x_norm[i];
		}
		z[o] = acc;
	}

	/*	causal depthwise conv: kernel tap 0 = oldest (t-3) .. tap 3 = current
	 * 	(t), then SiLU. conv_hist holds t-3,t-2,t-1 oldest-first. */
	for (int c = 0; c < SSM_D_INNER; c++) {
		const float *cw = &w->conv_w[c * SSM_D_CONV];
		float acc = w->conv_b[c];
		for (int k = 0; k < SSM_D_CONV - 1; k++) {
			acc += cw[k] * conv_hist[c][k];
		}
		acc += cw[SSM_D_CONV - 1] * u_raw[c];
		u[c] = ssm_silu(acc);

		for (int k = 0; k < SSM_D_CONV - 2; k++) {
			conv_hist[c][k] = conv_hist[c][k + 1];
		}
		conv_hist[c][SSM_D_CONV - 2] = u_raw[c];
	}

	/* x_proj on post-conv u -> delta_low / B / C */
	for (int o = 0; o < SSM_DT_RANK + 2 * SSM_D_STATE; o++) {
		float acc = 0.0f;
		const float *row = &w->x_proj_w[o * SSM_D_INNER];
		for (int i = 0; i < SSM_D_INNER; i++) {
			acc += row[i] * u[i];
		}
		x_dbl[o] = acc;
	}
	memcpy(delta_low, &x_dbl[0], sizeof(delta_low));
	memcpy(B, &x_dbl[SSM_DT_RANK], sizeof(B));
	memcpy(C, &x_dbl[SSM_DT_RANK + SSM_D_STATE], sizeof(C));

	/* dt_proj + softplus */
	for (int o = 0; o < SSM_D_INNER; o++) {
		float acc = w->dt_proj_b[o];
		const float *row = &w->dt_proj_w[o * SSM_DT_RANK];
		for (int i = 0; i < SSM_DT_RANK; i++) {
			acc += row[i] * delta_low[i];
		}
		delta[o] = ssm_softplus(acc);
	}

	/* 	euler discretize + recurrence + output, fused per timestep -- matches
	 * 	ssm_block.py's discretize(discretization="euler") +
	 * 	_scan_streaming(): A_bar = 1 + clamp(delta*A, min=-1.9),
	 * 	B_bar = delta*B. A_log is small enough (D_INNER*D_STATE floats) that
	 * 	recomputing exp() per step here is simpler than precomputing A once;
	 * 	the two D_MODEL x D_INNER matmuls above dominate per-frame cost. */
	for (int c = 0; c < SSM_D_INNER; c++) {
		float y_c = 0.0f;
		for (int n = 0; n < SSM_D_STATE; n++) {
			float A_cn = -expf(w->A_log[c * SSM_D_STATE + n]);
			float deltaA = delta[c] * A_cn;
			if (deltaA < -1.9f) {
				deltaA = -1.9f;
			}
			float A_bar = 1.0f + deltaA;
			float B_bar = delta[c] * B[n];
			float new_h = A_bar * h[c][n] + B_bar * u[c];
			h[c][n] = new_h;
			y_c += new_h * C[n];
		}
		y[c] = (y_c + u[c] * w->D[c]) * ssm_silu(z[c]);
	}

	for (int o = 0; o < SSM_D_MODEL; o++) {
		float acc = 0.0f;
		const float *row = &w->out_proj_w[o * SSM_D_INNER];
		for (int i = 0; i < SSM_D_INNER; i++) {
			acc += row[i] * y[i];
		}
		block_out[o] = acc;
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
		ssm_rmsnorm(x, ssm_blocks[layer].norm_w, x_norm);
		ssm_block_step(&ssm_blocks[layer], state->h[layer], state->conv_hist[layer], x_norm, block_out);
		for (int i = 0; i < SSM_D_MODEL; i++) {
			x[i] += block_out[i];
		}
	}

	float normed[SSM_D_MODEL];
	ssm_rmsnorm(x, ssm_final_norm_w, normed);

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
















