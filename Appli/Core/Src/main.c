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
#include "ssm_distance_head.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define EMBEDDING_RESPONSE_MAGIC 0x454D4245u  	/* "EMBE" */
#define HEAD_RESPONSE_MAGIC 0x44414548u        /* "HEAD" */
#define TIMING_RESPONSE_MAGIC 0x454D4954u      /* "TIME" */
#define UART_TX_CHUNK_SIZE 4096u

/* Upper bound on accepted clip length, in samples. Was the size of the
 * removed clip_buffer; now purely a sanity check on the stream header --
 * nothing this large is ever buffered. ~11 s at 16 kHz. */
#define AUDIO_MAX_SAMPLES 180000u
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

COM_InitTypeDef BspCOMInit;

DMA_HandleTypeDef handle_GPDMA1_Channel5;

/* USER CODE BEGIN PV */
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
static void MX_GPDMA1_Init(void);
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
	MX_GPDMA1_Init();
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
	/* Attach the GPDMA channel to COM1's UART handle. BSP_COM_Init() above
	 * initializes hcom_uart[COM1] independently of CubeMX/GPDMA -- this is
	 * the one line that connects the two, since CubeMX can't generate this
	 * link itself for a BSP-controlled peripheral. */
	__HAL_LINKDMA(&hcom_uart[COM1], hdmarx, handle_GPDMA1_Channel5);

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	while (1) {
		printf("Waiting for clip...\r\n");
		uint32_t num_hops = AudioIngest_ReceiveClipHeader(AUDIO_MAX_SAMPLES);

		if (num_hops == 0) {
			continue;
		}

		/* Clear any error flags latched during the header receive. An overrun
		 * (ORE) can set without failing that call, then cause the next
		 * HAL_UART_Receive to return HAL_ERROR immediately. */
		__HAL_UART_CLEAR_OREFLAG(&hcom_uart[COM1]);
		__HAL_UART_CLEAR_NEFLAG(&hcom_uart[COM1]);
		__HAL_UART_CLEAR_FEFLAG(&hcom_uart[COM1]);
		__HAL_UART_CLEAR_PEFLAG(&hcom_uart[COM1]);
		hcom_uart[COM1].ErrorCode = HAL_UART_ERROR_NONE;

		BSP_LED_On(LED_GREEN); /* solid: processing and responding */

		static SSMBackbone_State ssm_state __attribute__((section(".ssm_state_dtcm")));
		SSMBackbone_Reset(&ssm_state);

		static int16_t hop_buf[2][AUDIO_INGEST_HOP_SAMPLES] __attribute__((aligned(32)));
		uint32_t active = 0;

		uint32_t feature_cycles_total = 0;
		uint32_t normalize_cycles_total = 0;
		uint32_t backbone_cycles_total = 0;
		int rx_failed = 0;

		if (AudioIngest_StartHopReceive(hop_buf[0]) != HAL_OK) {
			rx_failed = 1;
		}

		for (uint32_t h = 0; h < num_hops && !rx_failed; h++) {
			if (AudioIngest_WaitHopComplete(hop_buf[active], 2000) != HAL_OK) {
				rx_failed = 1;
				break;
			}

			if (h + 1 < num_hops) {
				if (AudioIngest_StartHopReceive(hop_buf[1 - active])
						!= HAL_OK) {
					rx_failed = 1;
					break;
				}
			}

			float32_t hop_f32[AUDIO_INGEST_HOP_SAMPLES];
			for (uint32_t i = 0; i < AUDIO_INGEST_HOP_SAMPLES; i++) {
				hop_f32[i] = (float32_t) hop_buf[active][i] / 32768.0f;
			}

			uint32_t t0 = DWT->CYCCNT;
			FeaturePipeline_PushHop(hop_f32);
			float32_t logmel_frame[MEL_N_MELS];
			FeaturePipeline_ComputeLogMelFrame(
					FeaturePipeline_GetCurrentFrame(), logmel_frame);
			uint32_t t1 = DWT->CYCCNT;
			feature_cycles_total += (t1 - t0);

			float32_t normalized_frame[SSM_D_MODEL];
			SSMBackbone_NormalizeFrame(logmel_frame, normalized_frame);
			uint32_t t2 = DWT->CYCCNT;
			normalize_cycles_total += (t2 - t1);

			SSMBackbone_ProcessFrame(&ssm_state, normalized_frame, NULL);
			uint32_t t3 = DWT->CYCCNT;
			backbone_cycles_total += (t3 - t2);

			active = 1 - active;
		}

		if (rx_failed) {
			/* A hop failed to arrive, possibly mid-DMA-transfer. Without this,
			 * the UART is left marked busy, and every subsequent header
			 * receive fails instantly instead of timing out normally --
			 * producing an unthrottled retry loop instead of a paced one. */
			HAL_UART_AbortReceive(&hcom_uart[COM1]);
			BSP_LED_Off(LED_GREEN);
			printf("RXFAIL: hop receive failed, abandoning clip\r\n");
			continue;
		}

		static float32_t pooled_embedding[SSM_D_MODEL] __attribute__((aligned(32)));
		SSMBackbone_GetPooled(&ssm_state, pooled_embedding);

		/* Scheme-agnostic: SSMHeadResult's exact size differs between the
		 * float-only schemes (weight-only, weight+activation-boundaries)
		 * and true_int8 (which adds bonus int32 sumsq fields) -- this line
		 * doesn't need to know or care which one is linked in, since
		 * ssm_distance_head.h always declares the same struct name and
		 * function name (see mcu/ssm_head_src/ vs
		 * mcu/ssm_head_src_true_int8/'s shared public API). */
		static SSMHeadResult head_result __attribute__((aligned(32)));
		uint32_t t_head0 = DWT->CYCCNT;
		SSMDistanceHead_Score(pooled_embedding, &head_result);
		uint32_t head_cycles = DWT->CYCCNT - t_head0;

		uint8_t emb_header[8];
		emb_header[0] = (uint8_t) (EMBEDDING_RESPONSE_MAGIC & 0xFF);
		emb_header[1] = (uint8_t) ((EMBEDDING_RESPONSE_MAGIC >> 8) & 0xFF);
		emb_header[2] = (uint8_t) ((EMBEDDING_RESPONSE_MAGIC >> 16) & 0xFF);
		emb_header[3] = (uint8_t) ((EMBEDDING_RESPONSE_MAGIC >> 24) & 0xFF);
		emb_header[4] = (uint8_t) (SSM_D_MODEL & 0xFF);
		emb_header[5] = (uint8_t) ((SSM_D_MODEL >> 8) & 0xFF);
		emb_header[6] = 0;
		emb_header[7] = 0;

		if (HAL_UART_Transmit(&hcom_uart[COM1], emb_header, sizeof(emb_header),
				2000) != HAL_OK) {
			FatalBlink(LED_RED, 3);
		}
		SCB_CleanDCache_by_Addr((uint32_t*) pooled_embedding,
				(int32_t) ((SSM_D_MODEL * sizeof(float32_t) + 31u) & ~31u));
		if (TransmitAcked((uint8_t*) pooled_embedding,
		SSM_D_MODEL * sizeof(float32_t)) != HAL_OK) {
			FatalBlink(LED_RED, 1);
		}

		/* Second, separate packet: the head-scoring result. Deliberately
		 * additive, not a replacement for the embedding packet above --
		 * check_backbone_parity.py and every existing capture tool built
		 * against EMBEDDING_RESPONSE_MAGIC keeps working unchanged. Payload
		 * length is sizeof(head_result), read from this header rather than
		 * hardcoded, so a host-side parser can size its receive buffer
		 * correctly regardless of which scheme's SSMHeadResult (float-only
		 * vs true_int8's superset) this build was compiled against. */
		uint8_t head_header[8];
		head_header[0] = (uint8_t) (HEAD_RESPONSE_MAGIC & 0xFF);
		head_header[1] = (uint8_t) ((HEAD_RESPONSE_MAGIC >> 8) & 0xFF);
		head_header[2] = (uint8_t) ((HEAD_RESPONSE_MAGIC >> 16) & 0xFF);
		head_header[3] = (uint8_t) ((HEAD_RESPONSE_MAGIC >> 24) & 0xFF);
		head_header[4] = (uint8_t) (sizeof(head_result) & 0xFF);
		head_header[5] = (uint8_t) ((sizeof(head_result) >> 8) & 0xFF);
		head_header[6] = 0;
		head_header[7] = 0;

		if (HAL_UART_Transmit(&hcom_uart[COM1], head_header,
				sizeof(head_header), 2000) != HAL_OK) {
			FatalBlink(LED_RED, 3);
		}
		SCB_CleanDCache_by_Addr((uint32_t*) &head_result,
				(int32_t) ((sizeof(head_result) + 31u) & ~31u));
		if (TransmitAcked((uint8_t*) &head_result, sizeof(head_result))
				!= HAL_OK) {
			FatalBlink(LED_RED, 2);
		}

		/* Third, separate packet: per-clip timing. Fixed 4-uint32 layout
		 * regardless of scheme -- feature/normalize/backbone are already
		 * accumulated across every hop of this clip (matches how they're
		 * printed below); head_cycles is the single SSMDistanceHead_Score
		 * call timed just above, since it runs once per clip, not once
		 * per frame. */
		static uint32_t timing_values[4] __attribute__((aligned(32)));
		timing_values[0] = feature_cycles_total;
		timing_values[1] = normalize_cycles_total;
		timing_values[2] = backbone_cycles_total;
		timing_values[3] = head_cycles;

		uint8_t timing_header[8];
		timing_header[0] = (uint8_t) (TIMING_RESPONSE_MAGIC & 0xFF);
		timing_header[1] = (uint8_t) ((TIMING_RESPONSE_MAGIC >> 8) & 0xFF);
		timing_header[2] = (uint8_t) ((TIMING_RESPONSE_MAGIC >> 16) & 0xFF);
		timing_header[3] = (uint8_t) ((TIMING_RESPONSE_MAGIC >> 24) & 0xFF);
		timing_header[4] = (uint8_t) (sizeof(timing_values) & 0xFF);
		timing_header[5] = (uint8_t) ((sizeof(timing_values) >> 8) & 0xFF);
		timing_header[6] = 0;
		timing_header[7] = 0;

		if (HAL_UART_Transmit(&hcom_uart[COM1], timing_header,
				sizeof(timing_header), 2000) != HAL_OK) {
			FatalBlink(LED_RED, 3);
		}
		SCB_CleanDCache_by_Addr((uint32_t*) timing_values,
				(int32_t) ((sizeof(timing_values) + 31u) & ~31u));
		if (TransmitAcked((uint8_t*) timing_values, sizeof(timing_values))
				!= HAL_OK) {
			FatalBlink(LED_RED, 4);
		}

		BSP_LED_Off(LED_GREEN); /* done -- back to waiting for the next clip */

		printf(
				"hops=%lu feature=%lu cyc (%.2f ms) normalize=%lu cyc backbone=%lu cyc (%.2f ms) "
						"backbone/frame=%.0f cyc (%.1f us, budget 32000 us) head=%lu cyc "
						"euclidean=%.4f (%u) knn16=%.4f (%u)\r\n",
				(unsigned long) num_hops, (unsigned long) feature_cycles_total,
				(double) feature_cycles_total / SystemCoreClock * 1000.0,
				(unsigned long) normalize_cycles_total,
				(unsigned long) backbone_cycles_total,
				(double) backbone_cycles_total / SystemCoreClock * 1000.0,
				(double) backbone_cycles_total / num_hops,
				(double) backbone_cycles_total / num_hops / SystemCoreClock
						* 1e6, (unsigned long) head_cycles,
				(double) head_result.euclidean_score,
				(unsigned int) head_result.euclidean_anomaly,
				(double) head_result.knn16_score,
				(unsigned int) head_result.knn16_anomaly);
	}
	/* USER CODE END WHILE */

	/* USER CODE BEGIN 3 */
}
/* USER CODE END 3 */

/**
 * @brief GPDMA1 Initialization Function
 * @param None
 * @retval None
 */
static void MX_GPDMA1_Init(void) {

	/* USER CODE BEGIN GPDMA1_Init 0 */

	/* USER CODE END GPDMA1_Init 0 */

	/* Peripheral clock enable */
	__HAL_RCC_GPDMA1_CLK_ENABLE();

	/* GPDMA1 interrupt Init */
	HAL_NVIC_SetPriority(GPDMA1_Channel5_IRQn, 0, 0);
	HAL_NVIC_EnableIRQ(GPDMA1_Channel5_IRQn);

	/* USER CODE BEGIN GPDMA1_Init 1 */

	/* USER CODE END GPDMA1_Init 1 */
	handle_GPDMA1_Channel5.Instance = GPDMA1_Channel5;
	handle_GPDMA1_Channel5.Init.Request = DMA_REQUEST_SW;
	handle_GPDMA1_Channel5.Init.BlkHWRequest = DMA_BREQ_SINGLE_BURST;
	handle_GPDMA1_Channel5.Init.Direction = DMA_MEMORY_TO_MEMORY;
	handle_GPDMA1_Channel5.Init.SrcInc = DMA_SINC_FIXED;
	handle_GPDMA1_Channel5.Init.DestInc = DMA_DINC_FIXED;
	handle_GPDMA1_Channel5.Init.SrcDataWidth = DMA_SRC_DATAWIDTH_BYTE;
	handle_GPDMA1_Channel5.Init.DestDataWidth = DMA_DEST_DATAWIDTH_BYTE;
	handle_GPDMA1_Channel5.Init.Priority = DMA_LOW_PRIORITY_LOW_WEIGHT;
	handle_GPDMA1_Channel5.Init.SrcBurstLength = 1;
	handle_GPDMA1_Channel5.Init.DestBurstLength = 1;
	handle_GPDMA1_Channel5.Init.TransferAllocatedPort = DMA_SRC_ALLOCATED_PORT0
			| DMA_DEST_ALLOCATED_PORT0;
	handle_GPDMA1_Channel5.Init.TransferEventMode = DMA_TCEM_BLOCK_TRANSFER;
	handle_GPDMA1_Channel5.Init.Mode = DMA_NORMAL;
	if (HAL_DMA_Init(&handle_GPDMA1_Channel5) != HAL_OK) {
		Error_Handler();
	}
	if (HAL_DMA_ConfigChannelAttributes(&handle_GPDMA1_Channel5,
	DMA_CHANNEL_NPRIV) != HAL_OK) {
		Error_Handler();
	}
	/* USER CODE BEGIN GPDMA1_Init 2 */
	/* CubeMX cannot resolve a Request for USART3 (it's BSP-controlled, outside
	 * CubeMX's peripheral model), so the generated Init above defaults to a
	 * software-triggered memory-to-memory copy. Correct it here instead of
	 * editing the generated lines directly, since this block survives
	 * regeneration and those don't. */
	handle_GPDMA1_Channel5.Init.Request = GPDMA1_REQUEST_USART3_RX;
	handle_GPDMA1_Channel5.Init.Direction = DMA_PERIPH_TO_MEMORY;
	handle_GPDMA1_Channel5.Init.DestInc = DMA_DINC_INCREMENTED;
	if (HAL_DMA_Init(&handle_GPDMA1_Channel5) != HAL_OK) {
		Error_Handler();
	}
	/* USER CODE END GPDMA1_Init 2 */

}

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
