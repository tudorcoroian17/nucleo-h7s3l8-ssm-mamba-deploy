#ifndef FEATURE_PIPELINE_H
#define FEATURE_PIPELINE_H

#include "arm_math.h"
#include "mcu_feature_contract.h"
#include <stdbool.h>

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

/* Slides MEL_HOP_LENGTH new raw samples into the frame buffer, discarding
 * the oldest MEL_HOP_LENGTH samples. The buffer is zero-initialized at
 * startup, matching librosa.stft's pad_mode='constant' zero-padding for
 * center=True -- every frame produced, including the very first, is a
 * valid, correctly-padded frame. Nothing needs to be discarded. */
void FeaturePipeline_PushHop(const float32_t *new_hop_samples);

/* Returns a pointer to the current MEL_N_FFT-sample sliding frame, valid
 * until the next PushHop call. Feed this directly to ComputeLogMelFrame. */
const float32_t *FeaturePipeline_GetCurrentFrame(void);

#ifdef __cplusplus
}
#endif

#endif /* FEATURE_PIPELINE_H */
