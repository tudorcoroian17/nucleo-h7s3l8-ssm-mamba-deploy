#include "ssm_distance_head.h"

#include <math.h>
#include "ssm_head_ref.h"

static float ssm_l2_distance(const float *a, const float *b, int n) {
	float acc = 0.0f;
	for (int i = 0; i < n; i++) {
		float d = a[i] - b[i];
		acc += d * d;
	}
	return sqrtf(acc);
}

void SSMDistanceHead_Score(const float *embedding, SSMHeadResult *result) {
	result->euclidean_score = ssm_l2_distance(embedding, ssm_ref_centroid, SSM_D_MODEL);
	result->euclidean_anomaly = (result->euclidean_score > ssm_threshold_euclidean) ? 1 : 0;

	/* knn16 not scored in this folder -- sentinel. */
	result->knn16_score = -1.0f;
	result->knn16_anomaly = 0;
}
