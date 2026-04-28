/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "feature_provider.h"

#include "audio_provider.h"
#include "micro_features_micro_features_generator.h"
#include "micro_features_micro_model_settings.h"

extern int8_t g_feature_buffer[kFeatureElementCount];

namespace {
bool g_is_first_run = true;

TfLiteStatus PopulateFeatureDataImpl(tflite::ErrorReporter* error_reporter,
                                     int32_t last_time_in_ms,
                                     int32_t time_in_ms,
                                     int* how_many_new_slices) {
  // Quantize the time into steps as long as each window stride, so we can
  // figure out which audio data we need to fetch.
  const int last_step = (last_time_in_ms / kFeatureSliceStrideMs);
  const int current_step = (time_in_ms / kFeatureSliceStrideMs);

  int slices_needed = current_step - last_step;
  // If this is the first call, make sure we don't use any cached information.
  if (g_is_first_run) {
    TfLiteStatus init_status = InitializeMicroFeatures(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    g_is_first_run = false;
    slices_needed = kFeatureSliceCount;
  }
  if (slices_needed > kFeatureSliceCount) {
    slices_needed = kFeatureSliceCount;
  }
  *how_many_new_slices = slices_needed;

  const int slices_to_keep = kFeatureSliceCount - slices_needed;
  const int slices_to_drop = kFeatureSliceCount - slices_to_keep;
  if (slices_to_keep > 0) {
    for (int dest_slice = 0; dest_slice < slices_to_keep; ++dest_slice) {
      int8_t* dest_slice_data =
          g_feature_buffer + (dest_slice * kFeatureSliceSize);
      const int src_slice = dest_slice + slices_to_drop;
      const int8_t* src_slice_data =
          g_feature_buffer + (src_slice * kFeatureSliceSize);
      for (int i = 0; i < kFeatureSliceSize; ++i) {
        dest_slice_data[i] = src_slice_data[i];
      }
    }
  }

  if (slices_needed > 0) {
    for (int new_slice = slices_to_keep; new_slice < kFeatureSliceCount;
         ++new_slice) {
      const int duration_steps =
          kFeatureSliceDurationMs / kFeatureSliceStrideMs;
      const int first_step =
          current_step - (kFeatureSliceCount - 1) - duration_steps;
      const int new_step = first_step + new_slice;
      const int32_t slice_start_ms = (new_step * kFeatureSliceStrideMs);
      int16_t* audio_samples = nullptr;
      int audio_samples_size = 0;
      GetAudioSamples(error_reporter, (slice_start_ms > 0 ? slice_start_ms : 0),
                      kFeatureSliceDurationMs, &audio_samples_size,
                      &audio_samples);
      if (audio_samples_size < kMaxAudioSampleSize) {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Audio data size %d too small, want %d",
                             audio_samples_size, kMaxAudioSampleSize);
        return kTfLiteError;
      }
      int8_t* new_slice_data =
          g_feature_buffer + (new_slice * kFeatureSliceSize);
      size_t num_samples_read;
      TfLiteStatus generate_status = GenerateMicroFeatures(
          error_reporter, audio_samples, audio_samples_size, kFeatureSliceSize,
          new_slice_data, &num_samples_read);
      if (generate_status != kTfLiteOk) {
        return generate_status;
      }
    }
  }
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus InitializeFeatureProvider(tflite::ErrorReporter* error_reporter,
                                       int feature_size, int8_t* feature_data) {
  (void)feature_data;
  if (feature_size != kFeatureElementCount) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Requested feature_data_ size %d doesn't match %d",
                         feature_size, kFeatureElementCount);
    return kTfLiteError;
  }
  g_is_first_run = true;

  for (int n = 0; n < kFeatureElementCount; ++n) {
    g_feature_buffer[n] = 0;
  }
  return kTfLiteOk;
}

TfLiteStatus PopulateFeatureData(tflite::ErrorReporter* error_reporter,
                                 int32_t last_time_in_ms, int32_t time_in_ms,
                                 int* how_many_new_slices) {
  return PopulateFeatureDataImpl(error_reporter, last_time_in_ms, time_in_ms,
                                 how_many_new_slices);
}

FeatureProvider::FeatureProvider(int feature_size, int8_t* feature_data)
    : feature_size_(feature_size),
      feature_data_(feature_data),
      is_first_run_(true) {
  // Initialize the feature data to default values.
  for (int n = 0; n < feature_size_; ++n) {
    feature_data_[n] = 0;
  }
}

FeatureProvider::~FeatureProvider() {}

TfLiteStatus FeatureProvider::PopulateFeatureData(
    tflite::ErrorReporter* error_reporter, int32_t last_time_in_ms,
    int32_t time_in_ms, int* how_many_new_slices) {
  InitializeFeatureProvider(error_reporter, feature_size_, feature_data_);
  g_is_first_run = is_first_run_;
  TfLiteStatus status = PopulateFeatureDataImpl(
      error_reporter, last_time_in_ms, time_in_ms, how_many_new_slices);
  is_first_run_ = g_is_first_run;
  return status;
}
