#pragma once

#include <stdint.h>
#include "ssm_weights.h"

/* SINGLE-HEAD build: this folder scores euclidean only. The other head's
 * fields are set to a sentinel (score -1.0, anomaly 0) so this folder can
 * never be confused on the wire for its sibling that scores the other head
 * off the same backbone. SSMHeadResult keeps all four fields regardless, so
 * main.c is byte-for-byte identical across every setup. */

#define SSM_KNN16_N_CLUSTERS 16

typedef struct {
	float euclidean_score;
	float knn16_score;
	uint8_t euclidean_anomaly;
	uint8_t knn16_anomaly;
} SSMHeadResult;

void SSMDistanceHead_Score(const float *embedding, SSMHeadResult *result);
