/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2026 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include "feature_pipeline.h"
#include "audio_ingest.h"
#include "ssm_backbone.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define LOGMEL_RESPONSE_MAGIC 0x4C4D454Cu		/* "LMEL" */
#define EMBEDDING_RESPONSE_MAGIC 0x454D4245u  	/* "EMBE" */
#define LOGMEL_MAX_FRAMES ((AUDIO_INGEST_MAX_SAMPLES / MEL_HOP_LENGTH) + 1)
#define UART_TX_CHUNK_SIZE 4096u
#define UART_TX_CHUNK_DELAY_MS 150

#define SSM_SEND_LOGMEL_DEBUG 1  /* 0 once backbone parity is established */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

COM_InitTypeDef BspCOMInit;

/* USER CODE BEGIN PV */
static int16_t clip_buffer[AUDIO_INGEST_MAX_SAMPLES];
static float32_t logmel_output[LOGMEL_MAX_FRAMES][MEL_N_MELS] __attribute__((aligned(32)));
/* USER CODE END PV */
/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
static void FatalBlink(Led_TypeDef led, uint32_t count);
static HAL_StatusTypeDef TransmitAcked(const uint8_t *data,
		uint32_t total_bytes);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void) {

	/* USER CODE BEGIN 1 */

	/* USER CODE END 1 */

	/* Enable the CPU Cache */

	/* Enable I-Cache---------------------------------------------------------*/
	SCB_EnableICache();

	/* Enable D-Cache---------------------------------------------------------*/
	SCB_EnableDCache();

	/* MCU Configuration--------------------------------------------------------*/

	/* Update SystemCoreClock variable according to RCC registers values. */
	SystemCoreClockUpdate();

	/* Reset of all peripherals, Initializes the Flash interface and the Systick. */
	HAL_Init();

	/* USER CODE BEGIN Init */
	CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
	DWT->CYCCNT = 0;
	DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
	/* USER CODE END Init */

	/* USER CODE BEGIN SysInit */

	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	/* USER CODE BEGIN 2 */
	FeaturePipeline_Init();
	/* USER CODE END 2 */

	/* Initialize leds */
	BSP_LED_Init(LED_GREEN);
	BSP_LED_Init(LED_BLUE);
	BSP_LED_Init(LED_RED);

	/* Initialize USER push-button, will be used to trigger an interrupt each time it's pressed.*/
	BSP_PB_Init(BUTTON_USER, BUTTON_MODE_EXTI);

	/* Initialize COM1 port (115200, 8 bits (7-bit data + 1 stop bit), no parity */
	BspCOMInit.BaudRate = 115200;
	BspCOMInit.WordLength = COM_WORDLENGTH_8B;
	BspCOMInit.StopBits = COM_STOPBITS_1;
	BspCOMInit.Parity = COM_PARITY_NONE;
	BspCOMInit.HwFlowCtl = COM_HWCONTROL_NONE;
	if (BSP_COM_Init(COM1, &BspCOMInit) != BSP_ERROR_NONE) {
		Error_Handler();
	}

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	while (1) {
		printf("Waiting for clip...\r\n");
		uint32_t num_samples = AudioIngest_ReceiveClip(clip_buffer,
				AUDIO_INGEST_MAX_SAMPLES);

		if (num_samples == 0) {
			continue;
		}

		/* Clear any error flags latched during the long polled receive. An
		 * overrun (ORE) can set during the clip upload without failing that
		 * call, then cause the next HAL_UART_Receive to return HAL_ERROR
		 * immediately. */
		__HAL_UART_CLEAR_OREFLAG(&hcom_uart[COM1]);
		__HAL_UART_CLEAR_NEFLAG(&hcom_uart[COM1]);
		__HAL_UART_CLEAR_FEFLAG(&hcom_uart[COM1]);
		__HAL_UART_CLEAR_PEFLAG(&hcom_uart[COM1]);
		hcom_uart[COM1].ErrorCode = HAL_UART_ERROR_NONE;

		/* Drain any bytes the PC sent beyond what the clip receive consumed.
		 * These would otherwise be returned in place of the first ACK. */
		{
			uint8_t discard;
			while (HAL_UART_Receive(&hcom_uart[COM1], &discard, 1, 200)
					== HAL_OK) {
				/* keep reading until 200 ms passes with nothing arriving */
			}
			__HAL_UART_CLEAR_OREFLAG(&hcom_uart[COM1]);
			hcom_uart[COM1].ErrorCode = HAL_UART_ERROR_NONE;
		}

		BSP_LED_On(LED_GREEN); /* solid: processing and responding */

		static SSMBackbone_State ssm_state __attribute__((section(".ssm_state_dtcm")));
		SSMBackbone_Reset(&ssm_state);

		uint32_t feature_cycles_total = 0;
		uint32_t normalize_cycles_total = 0;
		uint32_t backbone_cycles_total = 0;

		uint32_t num_full_hops = num_samples / MEL_HOP_LENGTH;
		uint32_t leftover = num_samples - (num_full_hops * MEL_HOP_LENGTH);
		uint32_t num_frames = num_full_hops + 1;

		/* Compute every frame into RAM first. The actual transmission below
		 * is then one uninterrupted burst -- the same shape as the clip
		 * upload, which has been reliable every time -- instead of many
		 * small gapped bursts, which have not. */
		for (uint32_t h = 0; h < num_frames; h++) {
			float32_t hop_f32[MEL_HOP_LENGTH] = { 0 };

			if (h < num_full_hops) {
				for (uint32_t i = 0; i < MEL_HOP_LENGTH; i++) {
					hop_f32[i] = (float32_t) clip_buffer[h * MEL_HOP_LENGTH + i]
							/ 32768.0f;
				}
			} else {
				for (uint32_t i = 0; i < leftover; i++) {
					hop_f32[i] = (float32_t) clip_buffer[h * MEL_HOP_LENGTH + i]
							/ 32768.0f;
				}
			}

			uint32_t t0 = DWT->CYCCNT;
			FeaturePipeline_PushHop(hop_f32);
			FeaturePipeline_ComputeLogMelFrame(FeaturePipeline_GetCurrentFrame(), logmel_output[h]);
			uint32_t t1 = DWT->CYCCNT;
			feature_cycles_total += (t1 - t0);

			float32_t normalized_frame[SSM_D_MODEL];
			SSMBackbone_NormalizeFrame(logmel_output[h], normalized_frame);
			uint32_t t2 = DWT->CYCCNT;
			normalize_cycles_total += (t2 - t1);

			SSMBackbone_ProcessFrame(&ssm_state, normalized_frame, NULL);
			uint32_t t3 = DWT->CYCCNT;
			backbone_cycles_total += (t3 - t2);
		}

		static float32_t pooled_embedding[SSM_D_MODEL] __attribute__((aligned(32)));
		SSMBackbone_GetPooled(&ssm_state, pooled_embedding);

#if SSM_SEND_LOGMEL_DEBUG
		uint8_t resp_header[8];
		resp_header[0] = (uint8_t) (LOGMEL_RESPONSE_MAGIC & 0xFF);
		resp_header[1] = (uint8_t) ((LOGMEL_RESPONSE_MAGIC >> 8) & 0xFF);
		resp_header[2] = (uint8_t) ((LOGMEL_RESPONSE_MAGIC >> 16) & 0xFF);
		resp_header[3] = (uint8_t) ((LOGMEL_RESPONSE_MAGIC >> 24) & 0xFF);
		resp_header[4] = (uint8_t) (num_frames & 0xFF);
		resp_header[5] = (uint8_t) ((num_frames >> 8) & 0xFF);
		resp_header[6] = (uint8_t) ((num_frames >> 16) & 0xFF);
		resp_header[7] = (uint8_t) ((num_frames >> 24) & 0xFF);

		if (HAL_UART_Transmit(&hcom_uart[COM1], resp_header,
				sizeof(resp_header), 2000) != HAL_OK) {
			FatalBlink(LED_RED, 3);
		}

		uint32_t payload_bytes = num_frames * MEL_N_MELS * sizeof(float32_t);
		/* Flush the computed frames out of D-Cache to physical AXI SRAM before
		 * the UART reads them. Address and size must be 32-byte aligned. */
		SCB_CleanDCache_by_Addr((uint32_t*) logmel_output,
				(int32_t) ((payload_bytes + 31u) & ~31u));
		if (TransmitAcked((uint8_t*) logmel_output, payload_bytes) != HAL_OK) {
			FatalBlink(LED_RED, 1);
		}
#endif

		uint8_t emb_header[8];
		emb_header[0] = (uint8_t) (EMBEDDING_RESPONSE_MAGIC & 0xFF);
		emb_header[1] = (uint8_t) ((EMBEDDING_RESPONSE_MAGIC >> 8) & 0xFF);
		emb_header[2] = (uint8_t) ((EMBEDDING_RESPONSE_MAGIC >> 16) & 0xFF);
		emb_header[3] = (uint8_t) ((EMBEDDING_RESPONSE_MAGIC >> 24) & 0xFF);
		emb_header[4] = (uint8_t) (SSM_D_MODEL & 0xFF);
		emb_header[5] = (uint8_t) ((SSM_D_MODEL >> 8) & 0xFF);
		emb_header[6] = 0;
		emb_header[7] = 0;

		if (HAL_UART_Transmit(&hcom_uart[COM1], emb_header, sizeof(emb_header), 2000) != HAL_OK) {
		    FatalBlink(LED_RED, 3);
		}
		SCB_CleanDCache_by_Addr((uint32_t*) pooled_embedding,
		        (int32_t) ((SSM_D_MODEL * sizeof(float32_t) + 31u) & ~31u));
		if (TransmitAcked((uint8_t*) pooled_embedding, SSM_D_MODEL * sizeof(float32_t)) != HAL_OK) {
		    FatalBlink(LED_RED, 1);
		}

		BSP_LED_Off(LED_GREEN); /* done -- back to waiting for the next clip */

		printf("frames=%lu feature=%lu cyc (%.2f ms) normalize=%lu cyc backbone=%lu cyc (%.2f ms) "
		       "backbone/frame=%.0f cyc (%.1f us, budget 32000 us)\r\n",
		       (unsigned long) num_frames,
		       (unsigned long) feature_cycles_total,
		       (double) feature_cycles_total / SystemCoreClock * 1000.0,
		       (unsigned long) normalize_cycles_total,
		       (unsigned long) backbone_cycles_total,
		       (double) backbone_cycles_total / SystemCoreClock * 1000.0,
		       (double) backbone_cycles_total / num_frames,
		       (double) backbone_cycles_total / num_frames / SystemCoreClock * 1e6);
	}
	/* USER CODE END WHILE */

	/* USER CODE BEGIN 3 */
}
/* USER CODE END 3 */

/* USER CODE BEGIN 4 */
static void FatalBlink(Led_TypeDef led, uint32_t count) {
	while (1) {
		for (uint32_t i = 0; i < count; i++) {
			BSP_LED_On(led);
			HAL_Delay(200);
			BSP_LED_Off(led);
			HAL_Delay(200);
		}
		HAL_Delay(1000);
	}
}

static HAL_StatusTypeDef TransmitAcked(const uint8_t *data,
		uint32_t total_bytes) {
	uint32_t offset = 0;
	while (offset < total_bytes) {
		uint32_t chunk =
				(total_bytes - offset < UART_TX_CHUNK_SIZE) ?
						(total_bytes - offset) : UART_TX_CHUNK_SIZE;

		HAL_StatusTypeDef tx = HAL_UART_Transmit(&hcom_uart[COM1],
				(uint8_t*) &data[offset], chunk, 5000);
		if (tx != HAL_OK) {
			printf(
					"\r\nTXFAIL off=%lu chunk=%lu status=%d err=0x%08lX gState=0x%lX RxState=0x%lX\r\n",
					(unsigned long) offset, (unsigned long) chunk, (int) tx,
					(unsigned long) hcom_uart[COM1].ErrorCode,
					(unsigned long) hcom_uart[COM1].gState,
					(unsigned long) hcom_uart[COM1].RxState);
			return HAL_ERROR;
		}

		__HAL_UART_CLEAR_OREFLAG(&hcom_uart[COM1]);
		hcom_uart[COM1].ErrorCode = HAL_UART_ERROR_NONE;

		uint8_t ack[4];
		HAL_StatusTypeDef rx = HAL_UART_Receive(&hcom_uart[COM1], ack,
				sizeof(ack), 10000);
		if (rx != HAL_OK) {
			printf("\r\nACKFAIL off=%lu status=%d err=0x%08lX\r\n",
					(unsigned long) offset, (int) rx,
					(unsigned long) hcom_uart[COM1].ErrorCode);
			return HAL_ERROR;
		}
		if (ack[0] != 'A' || ack[1] != 'C' || ack[2] != 'K' || ack[3] != '!') {
			printf("\r\nACKBAD off=%lu bytes=%02X %02X %02X %02X\r\n",
					(unsigned long) offset, ack[0], ack[1], ack[2], ack[3]);
			return HAL_ERROR;
		}

		offset += chunk;
	}
	return HAL_OK;
}
/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void) {
	/* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1) {
	}
	/* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
