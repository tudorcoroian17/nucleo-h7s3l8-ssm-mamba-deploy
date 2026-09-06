#ifndef FEATURE_PIPELINE_H
#define FEATURE_PIPELINE_H

#include "arm_math.h"
#include "mcu_feature_contract.h"

#ifdef __cpluplus
extern "C" {
#endif

/* Call once, before any FeaturePipeline_ComputeLogMelFrame calls. */
void FeaturePipeline_Init(void);

/*
 * frame:      MEL_N_FFT (1024) raw audio samples, NOT yet windowed.
 * logmel_out: caller-provided buffer of MEL_N_MELS (64) floats. Receives
 *             natural-log mel energies, matching the Python reference's
 *             np.log(mel_spec + LOG_EPS).
 *
 * Not reentrant, not ISR-safe -- uses internal static scratch buffers.
 */
void FeaturePipeline_ComputeLogMelFrame(const float32_t *frame, float32_t *logmel_out);

#ifdef __cplusplus
}
#endif

#endif /* FEATURE_PIPELINE_H */
