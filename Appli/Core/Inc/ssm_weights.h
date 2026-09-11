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
    const int8_t *conv_b_q;
    const float *conv_b_scale;
    const int8_t *dt_proj_b_q;
    const float *dt_proj_b_scale;
    const int8_t *D_q;
    const float *D_scale;
    const int8_t *A_log_q;
    const float *A_log_scale;
    ssm_norm_weights_t norm_w;
} ssm_block_weights_t;

#define SSM_IN_PROJ_W(w, r, c) ((float)(w)->in_proj_w_q[(r) * SSM_D_MODEL + (c)] * (w)->in_proj_w_scale[0])
#define SSM_CONV_W(w, r, c) ((float)(w)->conv_w_q[(r) * SSM_D_CONV + (c)] * (w)->conv_w_scale[0])
#define SSM_X_PROJ_W(w, r, c) ((float)(w)->x_proj_w_q[(r) * SSM_D_INNER + (c)] * (w)->x_proj_w_scale[0])
#define SSM_DT_PROJ_W(w, r, c) ((float)(w)->dt_proj_w_q[(r) * SSM_DT_RANK + (c)] * (w)->dt_proj_w_scale[0])
#define SSM_OUT_PROJ_W(w, r, c) ((float)(w)->out_proj_w_q[(r) * SSM_D_INNER + (c)] * (w)->out_proj_w_scale[0])
#define SSM_CONV_B(w, i) ((float)(w)->conv_b_q[(i)] * (w)->conv_b_scale[0])
#define SSM_DT_PROJ_B(w, i) ((float)(w)->dt_proj_b_q[(i)] * (w)->dt_proj_b_scale[0])
#define SSM_D_PARAM(w, i) ((float)(w)->D_q[(i)] * (w)->D_scale[0])
#define SSM_A(w, r, c) (-expf((float)(w)->A_log_q[(r) * SSM_D_STATE + (c)] * (w)->A_log_scale[0]))
#define SSM_QUANT_U(w, val) (val)
#define SSM_QUANT_Z(w, val) (val)
#define SSM_QUANT_Y_GATED(w, val) (val)
#define SSM_QUANT_BLOCK_OUT(w, val) (val)
#define SSM_QUANT_FINAL_NORM(val) (val)
#define SSM_NORM_W(nw, i) ((float)(nw)->w_q[(i)] * (nw)->w_scale[0])

extern const ssm_block_weights_t ssm_blocks[SSM_N_LAYERS];
extern const ssm_norm_weights_t ssm_final_norm_w;


extern const float ssm_norm_mean[SSM_D_MODEL];
extern const float ssm_norm_std[SSM_D_MODEL];
