#include "feature_pipeline.h"
#include "main.h"
#include <math.h>

static arm_rfft_fast_instance_f32 rfft_instance;
static float32_t windowed_frame[MEL_N_FFT];
static float32_t fft_output[MEL_N_FFT];
static float32_t power_spectrum[MEL_N_FREQ_BINS];

void FeaturePipeline_Init(void) {
	if (ARM_MATH_SUCCESS != arm_rfft_fast_init_f32(&rfft_instance, MEL_N_FFT)) {
		Error_Handler();
	}
}

void FeaturePipeline_ComputeLogMelFrame(const float32_t *frame, float32_t *logmel_out) {
	/*	Window: elementwise multiply by the periodic Hann window that
	 *  matches librosa.stft(window='hann'). */
	arm_mult_f32((float32_t *)frame, (float32_t *)hann_window, windowed_frame, MEL_N_FFT);

	/*	Real FFT. CMSIS-DSP's "fast" RFFT packs the output as:
     *      fft_output[0]       = Re(X[0])      (DC bin -- always purely real)
     *      fft_output[1]       = Re(X[N/2])    (Nyquist bin -- always purely real)
     *      fft_output[2*k]     = Re(X[k])      for k = 1 .. N/2 - 1
     *      fft_output[2*k + 1] = Im(X[k])      for k = 1 .. N/2 - 1
     *  ifftFlag = 0 selects the forward transform. */
	arm_rfft_fast_f32(&rfft_instance, windowed_frame, fft_output, 0);

	/* 	Power spectrum (|X[k]|^2), matching np.abs(stft) ** 2. Bins 0 and
     *  N/2 are the packed real-only slots and must be handled separately
     *  from the ordinary complex bins in between -- treating fft_output[0]
     *  and fft_output[1] as one complex pair here would silently corrupt
     *  both the DC and Nyquist energy. */
	power_spectrum[0] = fft_output[0] * fft_output[0];
	power_spectrum[MEL_N_FFT / 2] = fft_output[1] * fft_output[1];
	arm_cmplx_mag_squared_f32(&fft_output[2], &power_spectrum[1], (MEL_N_FFT / 2) - 1);

	/*	Sparse mel matmul. Each mel bin touches only one contiguous run of
	 *  power_spectrum (findings/260) -- this replaces the full 64x513
	 *  dense mel_fb @ power_spec with 64 short dot products. */
	for (uint32_t mel = 0; mel < MEL_N_MELS; mel++) {
		float32_t energy;
		uint32_t start = mel_starts[mel];
		uint32_t length = mel_lengths[mel];
		uint32_t offset = mel_offsets[mel];

		arm_dot_prod_f32(&power_spectrum[start], &mel_values[offset], length, &energy);

		/*	Natural log with epsilon, matching np.log(mel_spec + LOG_EPS). */
		logmel_out[mel] = logf(energy + MEL_LOG_EPS);
	}
}
