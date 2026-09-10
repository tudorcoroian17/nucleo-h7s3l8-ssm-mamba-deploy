#pragma once

#include <stdint.h>
#include <math.h>

#define SSM_D_MODEL 64
#define SSM_D_STATE 16
#define SSM_D_INNER 64
#define SSM_D_CONV 4
#define SSM_DT_RANK 4
#define SSM_N_LAYERS 2

typedef struct {
    const int8_t *w_q;
    const float *w_scale;
} ssm_norm_weights_t;

typedef struct {
    const int8_t *in_proj_w_q;
    const float *in_proj_w_scale;
    const int8_t *conv_w_q;
    const float *conv_w_scale;
    const int8_t *x_proj_w_q;
    const float *x_proj_w_scale;
    const int8_t *dt_proj_w_q;
    const float *dt_proj_w_scale;
    const int8_t *out_proj_w_q;
    const float *out_proj_w_scale;
    const float *conv_b;
    const float *dt_proj_b;
    const float *D;
    const float *A;
    ssm_norm_weights_t norm_w;
} ssm_block_weights_t;

/* Weight access -- every field, quantized or not, goes through one of these.
 * Which branch got emitted (plain float, or int8+scale) depends only on
 * this export's --dtype/--weight-mode/--granularity; ssm_backbone.c's
 * source text never changes across schemes. */
#define SSM_IN_PROJ_W(w, r, c) ((float)(w)->in_proj_w_q[(r) * SSM_D_MODEL + (c)] * (w)->in_proj_w_scale[(r)])
#define SSM_CONV_W(w, r, c) ((float)(w)->conv_w_q[(r) * SSM_D_CONV + (c)] * (w)->conv_w_scale[(r)])
#define SSM_X_PROJ_W(w, r, c) ((float)(w)->x_proj_w_q[(r) * SSM_D_INNER + (c)] * (w)->x_proj_w_scale[(r)])
#define SSM_DT_PROJ_W(w, r, c) ((float)(w)->dt_proj_w_q[(r) * SSM_DT_RANK + (c)] * (w)->dt_proj_w_scale[(r)])
#define SSM_OUT_PROJ_W(w, r, c) ((float)(w)->out_proj_w_q[(r) * SSM_D_INNER + (c)] * (w)->out_proj_w_scale[(r)])
#define SSM_CONV_B(w, i) ((w)->conv_b[(i)])
#define SSM_DT_PROJ_B(w, i) ((w)->dt_proj_b[(i)])
#define SSM_D_PARAM(w, i) ((w)->D[(i)])
#define SSM_A(w, r, c) ((w)->A[(r) * SSM_D_STATE + (c)])
#define SSM_NORM_W(nw, i) ((float)(nw)->w_q[(i)] * (nw)->w_scale[0])

extern const ssm_block_weights_t ssm_blocks[SSM_N_LAYERS];
extern const ssm_norm_weights_t ssm_final_norm_w;

/* Per-channel z-score stats for this fold's training data (held-out case
 * 1). Apply as (raw - ssm_norm_mean) / ssm_norm_std
 * BEFORE feeding a log-mel frame into the backbone -- see
 * SSMBackbone_NormalizeFrame() in ssm_backbone.h. */
extern const float ssm_norm_mean[SSM_D_MODEL];
extern const float ssm_norm_std[SSM_D_MODEL];
