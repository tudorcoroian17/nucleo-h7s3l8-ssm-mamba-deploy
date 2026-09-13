#ifndef FEATURE_PIPELINE_H
#define FEATURE_PIPELINE_H

#include "arm_math.h"
#include "mcu_feature_contract.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cpluplus
extern "C" {
#endif

/* Call once, before any other FeaturePipeline call. */
void FeaturePipeline_Init(void);

/*
 * Assembles, scales and windows one frame from the retained overlap plus
 * new_hop, then retains new_hop as the overlap for the following frame.
 *
 * new_hop: MEL_HOP_LENGTH raw int16 samples, in the format the host sends.
 *          Scaling to float happens here -- the caller no longer converts.
 *
 * Once this returns, new_hop has been fully consumed and the caller may
 * reuse or overwrite it, including by starting a DMA into it. Nothing the
 * pipeline does after this point reads it.
 *
 * The overlap is zero-initialized at startup, matching librosa.stft's
 * pad_mode='constant' zero padding for center=True -- every frame produced,
 * including the first, is valid and correctly padded.
 *
 * Not reentrant, not ISR-safe -- uses internal static scratch.
 */
void FeaturePipeline_BeginFrame(const int16_t *new_hop);

/*
 * Transforms the frame assembled by the preceding FeaturePipeline_BeginFrame
 * call into log-mel energies.
 *
 * logmel_out: caller-provided buffer of MEL_N_MELS (64) floats. Receives
 *             natural-log mel energies, matching the Python reference's
 *             np.log(mel_spec + LOG_EPS).
 *
 * Calling this without a preceding BeginFrame produces garbage -- the
 * scratch buffer is whatever the previous frame left behind.
 */
void FeaturePipeline_FinishFrame(float32_t *logmel_out);

#ifdef __cplusplus
}
#endif

#endif /* FEATURE_PIPELINE_H */
