#pragma once

#define SSM_D_MODEL 64
#define SSM_D_STATE 16
#define SSM_D_INNER 64
#define SSM_D_CONV 4
#define SSM_DT_RANK 4
#define SSM_N_LAYERS 2

typedef struct {
    const float *in_proj_w;   /* [2*D_INNER][D_MODEL] */
    const float *conv_w;      /* [D_INNER][D_CONV] */
    const float *conv_b;      /* [D_INNER] */
    const float *x_proj_w;    /* [DT_RANK+2*D_STATE][D_INNER] */
    const float *dt_proj_w;   /* [D_INNER][DT_RANK] */
    const float *dt_proj_b;   /* [D_INNER] */
    const float *A_log;       /* [D_INNER][D_STATE] */
    const float *D;           /* [D_INNER] */
    const float *out_proj_w;  /* [D_MODEL][D_INNER] */
    const float *norm_w;      /* [D_MODEL] */
} ssm_block_weights_t;

extern const ssm_block_weights_t ssm_blocks[SSM_N_LAYERS];
extern const float ssm_final_norm_w[SSM_D_MODEL];

/* Per-channel z-score stats for this fold's training data (held-out case
 * 1). Apply as (raw - ssm_norm_mean) / ssm_norm_std
 * BEFORE feeding a log-mel frame into the backbone -- see
 * SSMBackbone_NormalizeFrame() in ssm_backbone.h. */
extern const float ssm_norm_mean[SSM_D_MODEL];
extern const float ssm_norm_std[SSM_D_MODEL];
