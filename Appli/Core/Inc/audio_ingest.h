#ifndef AUDIO_INGEST_H
#define AUDIO_INGEST_H

#include <stdint.h>

#define AUDIO_INGEST_MAGIC 0x41554449u /* "AUDI */
/* Sized for one ~11 s ToyCar/ToyTrain-length clip at 16 kHz (matching the
 * (344, 64) shape logmel.py's own smoke test expects). This is a working
 * default, not a verified RAM budget -- that's still open (Phase 3 §3.6). */
#define AUDIO_INGEST_MAX_SAMPLES 180000u

/* Blocks waiting for one clip over UART: reads the magic, the sample count,
 * then that many int16 PCM samples. Returns the actual sample count
 * received, or 0 on a magic mismatch, an oversized count, or a UART error.
 * Do not call printf while this is in progress -- it shares the same UART
 * as the debug console, and the two would interleave and corrupt each
 * other. */
uint32_t AudioIngest_ReceiveClip(int16_t *out_buffer, uint32_t max_samples);

#endif /* AUDIO_INGEST_H */
