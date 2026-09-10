#include "audio_ingest.h"
#include "main.h"
#include "stm32h7rsxx_nucleo.h"

#define UART_RX_TIMEOUT_MS 5000

static volatile uint8_t hop_rx_complete = 0;
static volatile uint8_t hop_rx_error = 0;

uint32_t AudioIngest_ReceiveClipHeader(uint32_t max_samples) {
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

	uint32_t num_hops = (num_samples + AUDIO_INGEST_HOP_SAMPLES - 1u)
			/ AUDIO_INGEST_HOP_SAMPLES;

	return num_hops;
}

HAL_StatusTypeDef AudioIngest_StartHopReceive(int16_t *out_hop) {
	hop_rx_complete = 0;
	hop_rx_error = 0;
	return HAL_UART_Receive_DMA(&hcom_uart[COM1], (uint8_t*) out_hop,
			AUDIO_INGEST_HOP_SAMPLES * sizeof(int16_t));
}

HAL_StatusTypeDef AudioIngest_WaitHopComplete(int16_t *out_hop,
		uint32_t timeout_ms) {
	uint32_t start = HAL_GetTick();
	while (!hop_rx_complete && !hop_rx_error) {
		if ((HAL_GetTick() - start) > timeout_ms) {
			return HAL_TIMEOUT;
		}
	}
	if (hop_rx_error) {
		return HAL_ERROR;
	}

	SCB_InvalidateDCache_by_Addr((uint32_t*) out_hop,
			(int32_t) (AUDIO_INGEST_HOP_SAMPLES * sizeof(int16_t)));

	return HAL_OK;
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
	if (huart == &hcom_uart[COM1]) {
		hop_rx_complete = 1;
	}
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart) {
	if (huart == &hcom_uart[COM1]) {
		hop_rx_error = 1;
	}
}
