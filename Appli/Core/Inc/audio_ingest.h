#ifndef AUDIO_INGEST_H
#define AUDIO_INGEST_H

#include <stdint.h>
#include "stm32h7rsxx_hal.h"

#define AUDIO_INGEST_MAGIC 0x41554449u /* "AUDI" */
#define AUDIO_INGEST_HOP_SAMPLES 512u

/* Blocks waiting for one clip's stream header over UART: reads the magic,
 * then the total sample count. Returns the number of hops the host will
 * send for this clip (ceil(num_samples / AUDIO_INGEST_HOP_SAMPLES)), or 0 on
 * a magic mismatch, a zero/oversized sample count, or a UART error.
 *
 * max_samples bounds the accepted sample count; a clip larger than this is
 * rejected (returns 0) rather than truncated. */
uint32_t AudioIngest_ReceiveClipHeader(uint32_t max_samples);

/* Starts a DMA receive of exactly AUDIO_INGEST_HOP_SAMPLES int16 samples
 * into out_hop. Non-blocking -- returns as soon as the DMA transfer is
 * queued, well before the data has arrived. out_hop must stay valid and
 * must not be touched until AudioIngest_WaitHopComplete() returns HAL_OK
 * for this call, and must be 32-byte aligned (cache-line size on this
 * part) for the invalidate step in AudioIngest_WaitHopComplete() to be
 * safe.
 *
 * Only one hop receive may be in flight at a time -- do not call this
 * again before the previous call's AudioIngest_WaitHopComplete() has
 * returned. */
HAL_StatusTypeDef AudioIngest_StartHopReceive(int16_t *out_hop);

/* Blocks until the most recently started AudioIngest_StartHopReceive()
 * call finishes, invalidating D-Cache over the received buffer so the CPU
 * sees the DMA-written data rather than a stale cached copy. Returns
 * HAL_OK on success, HAL_TIMEOUT if timeout_ms elapses first, or HAL_ERROR
 * if the UART reported a receive error (framing, overrun, noise). */
HAL_StatusTypeDef AudioIngest_WaitHopComplete(int16_t *out_hop, uint32_t timeout_ms);

#endif /* AUDIO_INGEST_H */
