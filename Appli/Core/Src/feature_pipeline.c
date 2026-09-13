#include "feature_pipeline.h"
#include "main.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

/*	Frame assembly below writes the overlap into the first half of
 *  windowed_frame and the incoming hop into the second half. That split is
 *  only correct at exactly 50% overlap. An off-by-one at this boundary
 *  produces a subtly wrong spectrum rather than a crash, so the assumption
 *  is checked at compile time instead of being left to a comment. */
_Static_assert(MEL_N_FFT == 2 * MEL_HOP_LENGTH,
		"frame assembly assumes MEL_HOP_LENGTH == MEL_N_FFT / 2");

static arm_rfft_fast_instance_f32 rfft_instance;
static float32_t windowed_frame[MEL_N_FFT];
static float32_t fft_output[MEL_N_FFT];

/*	The power spectrum shares storage with windowed_frame. By the time the
 *  magnitudes are written, arm_rfft_fast_f32 has consumed windowed_frame --
 *  and in fact overwritten it, since the fast RFFT uses its input buffer as
 *  CFFT scratch -- so those 1024 floats are dead. 513 are needed here.
 *
 *  This aliases storage, not call arguments: at every CMSIS call site the
 *  source and destination are still distinct arrays, so nothing depends on
 *  in-place CMSIS behaviour. That distinction matters -- arm_rfft_fast_f32
 *  was measured to corrupt 510 of 1024 outputs when called with pSrc ==
 *  pDst (plans/detailed/530, step 4). */
static float32_t *const power_spectrum = windowed_frame;

/*	The previous hop, kept in the wire format rather than as floats. This is
 *  the pipeline's only state between frames: at 50% overlap the first half
 *  of frame N is exactly the hop received for frame N-1, so a full-frame
 *  float history was storing 4096 bytes to hold 1024 bytes of information.
 *  Zero-initialized: static storage duration. */
static int16_t overlap[MEL_HOP_LENGTH];

void FeaturePipeline_Init(void) {
	if (ARM_MATH_SUCCESS != arm_rfft_fast_init_1024_f32(&rfft_instance)) {
		Error_Handler();
	}
}

void FeaturePipeline_BeginFrame(const int16_t *new_hop) {
	/*	Assemble, scale and window in one pass. The old pipeline did this in
	 *  three: convert the hop to floats, memmove a 1024-float history down
	 *  by one hop, then arm_mult_f32 the whole frame by the window. Fusing
	 *  them removes the float history and the caller's conversion buffer.
	 *
	 *  Bit-exact against the old path: it computed (s / 32768.0f), stored
	 *  it, and later multiplied by hann_window[i]. Same two operations, same
	 *  order, same operands. 32768 is a power of two, so the scaling is
	 *  exact and the window multiply is the only rounding, as before. Do not
	 *  "simplify" this to s * (hann_window[i] / 32768.0f) -- that reassociates
	 *  and changes the result. */
	for (uint32_t i = 0; i < MEL_HOP_LENGTH; i++) {
		windowed_frame[i] = ((float32_t) overlap[i] / 32768.0f)
				* hann_window[i];
	}
	for (uint32_t i = 0; i < MEL_HOP_LENGTH; i++) {
		windowed_frame[MEL_HOP_LENGTH + i] =
				((float32_t) new_hop[i] / 32768.0f)
						* hann_window[MEL_HOP_LENGTH + i];
	}

	/*	new_hop becomes the overlap for the next frame. After this line the
	 *  caller's buffer is free, which is what will let the DMA for the next
	 *  hop start before the FFT runs (step 3). */
	memcpy(overlap, new_hop, MEL_HOP_LENGTH * sizeof(int16_t));
}

void FeaturePipeline_FinishFrame(float32_t *logmel_out) {
	/*	Real FFT. CMSIS-DSP's "fast" RFFT packs the output as:
	 *      fft_output[0]       = Re(X[0])      (DC bin -- always purely real)
	 *      fft_output[1]       = Re(X[N/2])    (Nyquist bin -- always purely real)
	 *      fft_output[2*k]     = Re(X[k])      for k = 1 .. N/2 - 1
	 *      fft_output[2*k + 1] = Im(X[k])      for k = 1 .. N/2 - 1
	 *  ifftFlag = 0 selects the forward transform.
	 *
	 *  This call is also what kills windowed_frame, freeing it to back
	 *  power_spectrum below. Do not read windowed_frame after this line. */
	arm_rfft_fast_f32(&rfft_instance, windowed_frame, fft_output, 0);

	/* 	Power spectrum (|X[k]|^2), matching np.abs(stft) ** 2. Bins 0 and
	 *  N/2 are the packed real-only slots and must be handled separately
	 *  from the ordinary complex bins in between -- treating fft_output[0]
	 *  and fft_output[1] as one complex pair here would silently corrupt
	 *  both the DC and Nyquist energy. */
	power_spectrum[0] = fft_output[0] * fft_output[0];
	power_spectrum[MEL_N_FFT / 2] = fft_output[1] * fft_output[1];
	arm_cmplx_mag_squared_f32(&fft_output[2], &power_spectrum[1],
			(MEL_N_FFT / 2) - 1);

	/*	Sparse mel matmul. Each mel bin touches only one contiguous run of
	 *  power_spectrum (findings/260) -- this replaces the full 64x513
	 *  dense mel_fb @ power_spec with 64 short dot products. */
	for (uint32_t mel = 0; mel < MEL_N_MELS; mel++) {
		float32_t energy;
		uint32_t start = mel_starts[mel];
		uint32_t length = mel_lengths[mel];
		uint32_t offset = mel_offsets[mel];

		arm_dot_prod_f32(&power_spectrum[start], &mel_values[offset], length,
				&energy);

		/*	Natural log with epsilon, matching np.log(mel_spec + LOG_EPS). */
		logmel_out[mel] = logf(energy + MEL_LOG_EPS);
	}
}
