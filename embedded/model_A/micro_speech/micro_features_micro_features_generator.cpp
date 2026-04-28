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

#include "micro_features_micro_model_settings.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"

namespace {

FrontendState g_micro_features_state;
float g_input_scale = 1.0f;
int g_input_zero_point = 0;
bool g_is_initialized = false;
bool g_is_first_time = true;

int8_t QuantizeFrontendValue(uint16_t value) {
  const float scaled = (static_cast<float>(value) / g_input_scale) +
                       static_cast<float>(g_input_zero_point);
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

  FrontendConfig config;
  config.window.size_ms = kFeatureSliceDurationMs;
  config.window.step_size_ms = kFeatureSliceStrideMs;
  config.noise_reduction.smoothing_bits = 10;
  config.filterbank.num_channels = kFeatureSliceSize;
  config.filterbank.lower_band_limit = 125.0;
  config.filterbank.upper_band_limit = 7500.0;
  config.noise_reduction.even_smoothing = 0.025;
  config.noise_reduction.odd_smoothing = 0.06;
  config.noise_reduction.min_signal_remaining = 0.05;
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength = 0.95;
  config.pcan_gain_control.offset = 80.0;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;

  if (!FrontendPopulateState(&config, &g_micro_features_state,
                             kAudioSampleFrequency)) {
    TF_LITE_REPORT_ERROR(error_reporter, "FrontendPopulateState() failed");
    return kTfLiteError;
  }

  g_is_initialized = true;
  g_is_first_time = true;
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
  if (input_size < kMaxAudioSampleSize) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Audio data size %d too small, want %d", input_size,
                         kMaxAudioSampleSize);
    return kTfLiteError;
  }
  if (output_size != kFeatureSliceSize) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Output size %d doesn't match frontend channel count %d",
                         output_size, kFeatureSliceSize);
    return kTfLiteError;
  }

  const int16_t* frontend_input = input;
  int frontend_input_size = input_size;
  if (!g_is_first_time) {
    const int overlap_ms = kFeatureSliceDurationMs - kFeatureSliceStrideMs;
    const int overlap_samples = (overlap_ms * kAudioSampleFrequency) / 1000;
    frontend_input += overlap_samples;
    frontend_input_size -= overlap_samples;
  }
  g_is_first_time = false;

  FrontendOutput frontend_output = FrontendProcessSamples(
      &g_micro_features_state, frontend_input, frontend_input_size,
      num_samples_read);
  if (frontend_output.size != kFeatureSliceSize) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Frontend output size %d doesn't match %d",
                         frontend_output.size, kFeatureSliceSize);
    return kTfLiteError;
  }

  for (size_t i = 0; i < frontend_output.size; ++i) {
    output[i] = QuantizeFrontendValue(frontend_output.values[i]);
  }

  return kTfLiteOk;
}
