/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "micro_features_micro_features_generator.h"

#include <math.h>

#include "micro_features_micro_model_settings.h"

namespace {

constexpr int kFftLength = 256;
constexpr int kFftBinCount = (kFftLength / 2) + 1;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kInt16ToFloat = 1.0f / 32768.0f;

float g_hann_window[kFftLength];
float g_twiddle_real[kFftLength / 2];
float g_twiddle_imag[kFftLength / 2];
float g_fft_real[kFftLength];
float g_fft_imag[kFftLength];
float g_input_scale = 1.0f;
int g_input_zero_point = 0;
bool g_is_initialized = false;

int ReverseBits(int value, int bit_count) {
  int reversed = 0;
  for (int i = 0; i < bit_count; ++i) {
    reversed <<= 1;
    reversed |= value & 1;
    value >>= 1;
  }
  return reversed;
}

void Fft256(float* real, float* imag) {
  constexpr int kBitCount = 8;
  for (int i = 0; i < kFftLength; ++i) {
    const int j = ReverseBits(i, kBitCount);
    if (j > i) {
      const float real_temp = real[i];
      const float imag_temp = imag[i];
      real[i] = real[j];
      imag[i] = imag[j];
      real[j] = real_temp;
      imag[j] = imag_temp;
    }
  }

  for (int fft_size = 2; fft_size <= kFftLength; fft_size <<= 1) {
    const int half_size = fft_size / 2;
    const int table_step = kFftLength / fft_size;
    for (int group = 0; group < kFftLength; group += fft_size) {
      for (int pair = 0; pair < half_size; ++pair) {
        const int table_index = pair * table_step;
        const float twiddle_real = g_twiddle_real[table_index];
        const float twiddle_imag = g_twiddle_imag[table_index];
        const int even_index = group + pair;
        const int odd_index = even_index + half_size;

        const float odd_real = real[odd_index];
        const float odd_imag = imag[odd_index];
        const float temp_real =
            (twiddle_real * odd_real) - (twiddle_imag * odd_imag);
        const float temp_imag =
            (twiddle_real * odd_imag) + (twiddle_imag * odd_real);

        real[odd_index] = real[even_index] - temp_real;
        imag[odd_index] = imag[even_index] - temp_imag;
        real[even_index] += temp_real;
        imag[even_index] += temp_imag;
      }
    }
  }
}

int8_t QuantizeMagnitude(float value) {
  const float scaled = (value / g_input_scale) + g_input_zero_point;
  int quantized =
      static_cast<int>((scaled >= 0.0f) ? (scaled + 0.5f) : (scaled - 0.5f));
  if (quantized < -128) {
    quantized = -128;
  }
  if (quantized > 127) {
    quantized = 127;
  }
  return static_cast<int8_t>(quantized);
}

}  // namespace

TfLiteStatus InitializeMicroFeatures(tflite::ErrorReporter* error_reporter) {
  if (g_input_scale <= 0.0f) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Input quantization scale must be positive");
    return kTfLiteError;
  }

  for (int i = 0; i < kFftLength; ++i) {
    // tf.signal.stft uses tf.signal.hann_window(periodic=True) by default.
    g_hann_window[i] =
        0.5f - (0.5f * cosf((2.0f * kPi * i) / kFftLength));
  }
  for (int i = 0; i < (kFftLength / 2); ++i) {
    const float angle = (-2.0f * kPi * i) / kFftLength;
    g_twiddle_real[i] = cosf(angle);
    g_twiddle_imag[i] = sinf(angle);
  }

  g_is_initialized = true;
  return kTfLiteOk;
}

void SetMicroFeaturesInputQuantization(float scale, int zero_point) {
  g_input_scale = scale;
  g_input_zero_point = zero_point;
}

TfLiteStatus GenerateMicroFeatures(tflite::ErrorReporter* error_reporter,
                                   const int16_t* input, int input_size,
                                   int output_size, int8_t* output,
                                   size_t* num_samples_read) {
  if (!g_is_initialized) {
    TfLiteStatus init_status = InitializeMicroFeatures(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
  }
  if (input_size < kFftLength) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Audio data size %d too small, want %d", input_size,
                         kFftLength);
    return kTfLiteError;
  }
  if (output_size != kFftBinCount) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Output size %d doesn't match STFT bin count %d",
                         output_size, kFftBinCount);
    return kTfLiteError;
  }

  for (int i = 0; i < kFftLength; ++i) {
    g_fft_real[i] =
        static_cast<float>(input[i]) * kInt16ToFloat * g_hann_window[i];
    g_fft_imag[i] = 0.0f;
  }

  Fft256(g_fft_real, g_fft_imag);

  for (int i = 0; i < kFftBinCount; ++i) {
    const float magnitude =
        sqrtf((g_fft_real[i] * g_fft_real[i]) +
              (g_fft_imag[i] * g_fft_imag[i]));
    output[i] = QuantizeMagnitude(magnitude);
  }

  *num_samples_read = kFftLength;
  return kTfLiteOk;
}
