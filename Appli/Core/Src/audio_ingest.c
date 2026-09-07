#include "audio_ingest.h"
#include "main.h"
#include "stm32h7rsxx_nucleo.h"

#define UART_RX_TIMEOUT_MS 5000
#define UART_RX_CHUNK_SIZE 4096u

static HAL_StatusTypeDef ReceiveChunked(uint8_t *dest, uint32_t total_bytes,
		uint32_t chunk_timeout_ms) {
	uint32_t offset = 0;
	while (offset < total_bytes) {
		uint32_t chunk =
				(total_bytes - offset < UART_RX_CHUNK_SIZE) ?
						(total_bytes - offset) : UART_RX_CHUNK_SIZE;

		/* HAL's Size parameter is uint16_t. Any single call above 65535 bytes
		 * silently truncates modulo 65536 and still returns HAL_OK. */
		if (HAL_UART_Receive(&hcom_uart[COM1], &dest[offset], (uint16_t) chunk,
				chunk_timeout_ms) != HAL_OK) {
			return HAL_ERROR;
		}
		offset += chunk;
	}
	return HAL_OK;
}

uint32_t AudioIngest_ReceiveClip(int16_t *out_buffer, uint32_t max_samples) {
	uint8_t header[8];

	if (HAL_UART_Receive(&hcom_uart[COM1], header, sizeof(header),
			UART_RX_TIMEOUT_MS) != HAL_OK) {
		return 0;
	}

	uint32_t magic = (uint32_t) header[0] | ((uint32_t) header[1] << 8)
			| ((uint32_t) header[2] << 16) | ((uint32_t) header[3] << 24);
	if (magic != AUDIO_INGEST_MAGIC) {
		return 0;
	}

	uint32_t num_samples = (uint32_t) header[4] | ((uint32_t) header[5] << 8)
			| ((uint32_t) header[6] << 16) | ((uint32_t) header[7] << 24);
	if (num_samples == 0 || num_samples > max_samples) {
		return 0;
	}

	/*	Cortex-M7 on this board is little-endian, matching the wire format, so
	 *	a direct byte-for-byte receive into the int16_t buffer is correct
	 *	as-is -- no byte-swapping needed. */
	if (ReceiveChunked((uint8_t*) out_buffer, num_samples * sizeof(int16_t),
			UART_RX_TIMEOUT_MS) != HAL_OK) {
		return 0;
	}

	return num_samples;
}
