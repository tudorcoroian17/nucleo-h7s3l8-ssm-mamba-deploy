#pragma once

#include "ssm_weights.h"

#define SSM_KNN16_N_CLUSTERS 16

extern const float ssm_ref_centroid[SSM_D_MODEL];
extern const float ssm_ref_clusters[SSM_KNN16_N_CLUSTERS][SSM_D_MODEL];
extern const float ssm_threshold_euclidean;
extern const float ssm_threshold_knn16;
