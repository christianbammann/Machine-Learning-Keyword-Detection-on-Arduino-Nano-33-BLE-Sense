/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <TensorFlowLite.h>

#include "Arduino.h"
#include "main_functions.h"

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "local_strided_slice.h"
#include "micro_features_micro_features_generator.h"
#include "micro_features_micro_model_settings.h"
#include "micro_features_model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#define private public
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#undef private
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

int8_t g_feature_buffer[kFeatureElementCount];

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;
bool is_setup_complete = false;
constexpr bool kEnableSetupDebug = false;
constexpr bool kEnableInferenceDebug = false;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 120 * 1024;  // Increase if AllocateTensors fails.
uint8_t tensor_arena[kTensorArenaSize];
int8_t* model_input_buffer = nullptr;

bool IsExpectedInputShape(const TfLiteTensor* tensor) {
  if ((tensor->dims->size == 2) && (tensor->dims->data[0] == 1) &&
      (tensor->dims->data[1] == kFeatureElementCount)) {
    return true;
  }
  if ((tensor->dims->size == 3) && (tensor->dims->data[0] == 1) &&
      (tensor->dims->data[1] == kFeatureSliceCount) &&
      (tensor->dims->data[2] == kFeatureSliceSize)) {
    return true;
  }
  if ((tensor->dims->size == 4) && (tensor->dims->data[0] == 1) &&
      (tensor->dims->data[1] == kFeatureSliceCount) &&
      (tensor->dims->data[2] == kFeatureSliceSize) &&
      (tensor->dims->data[3] == 1)) {
    return true;
  }
  return false;
}
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  Serial.begin(115200);
  const int32_t serial_start = millis();
  while (!Serial && ((millis() - serial_start) < 3000)) {
  }
  if (kEnableSetupDebug) {
    Serial.println("setup: start");
  }

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  if (kEnableSetupDebug) {
    Serial.println("setup: error reporter ready");
  }

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  if (kEnableSetupDebug) {
    Serial.println("setup: model loaded");
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<8> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AddFullyConnected() failed");
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AddSoftmax() failed");
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AddReshape() failed");
    return;
  }
  if (micro_op_resolver.AddShape() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AddShape() failed");
    return;
  }
  if (micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_STRIDED_SLICE,
                                   RegisterLocalStridedSlice(),
                                   tflite::ParseStridedSlice) != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AddStridedSlice() failed");
    return;
  }
  if (micro_op_resolver.AddPack() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AddPack() failed");
    return;
  }
  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AddMaxPool2D() failed");
    return;
  }
  if (micro_op_resolver.AddConv2D() != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AddConv2D() failed");
    return;
  }
  if (kEnableSetupDebug) {
    Serial.println("setup: ops registered");
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    if (kEnableSetupDebug) {
      Serial.println("setup: AllocateTensors failed");
    }
    return;
  }
  if (kEnableSetupDebug) {
    Serial.println("setup: tensors allocated");
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if (!IsExpectedInputShape(model_input) || (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    if (kEnableSetupDebug) {
      Serial.println("setup: bad input tensor");
    }
    return;
  }
  model_input_buffer = model_input->data.int8;
  SetMicroFeaturesInputQuantization(model_input->params.scale,
                                    model_input->params.zero_point);
  if (kEnableSetupDebug) {
    Serial.println("setup: input ready");
  }

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  TfLiteStatus feature_provider_status =
      InitializeFeatureProvider(error_reporter, kFeatureElementCount,
                                g_feature_buffer);
  if (feature_provider_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature provider setup failed");
    if (kEnableSetupDebug) {
      Serial.println("setup: feature provider failed");
    }
    return;
  }
  if (kEnableSetupDebug) {
    Serial.println("setup: feature provider ready");
  }

  static RecognizeCommands static_recognizer(
      error_reporter,
      0,     // Use the latest inference only; this model runs slower than 1 Hz.
      160,   // Detection threshold on dequantized 0-255 scores.
      1000,  // Suppression time after a command fires.
      1);    // Only require one inference result.
  recognizer = &static_recognizer;

  previous_time = 0;
  is_setup_complete = true;
  if (kEnableSetupDebug) {
    TF_LITE_REPORT_ERROR(error_reporter, "micro_speech setup complete");
    Serial.println("setup: complete");
  }
}

// The name of this function is important for Arduino compatibility.
void loop() {
  if (!is_setup_complete) {
    return;
  }

  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }
  previous_time = current_time;

  static int32_t last_audio_debug_time = 0;
  if (kEnableInferenceDebug &&
      ((current_time - last_audio_debug_time) >= 1000)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Audio @%dms slices=%d",
                         current_time, how_many_new_slices);
    last_audio_debug_time = current_time;
  }

  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }

  // Copy feature buffer to input tensor.
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = g_feature_buffer[i];
  }

  // Run the model on the spectrogram input and make sure it succeeds.
  const uint32_t invoke_start = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  const uint32_t invoke_time = millis() - invoke_start;
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }
  if (kEnableInferenceDebug) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke ms: %d", invoke_time);
  }

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);

  static int32_t last_raw_debug_time = 0;
  if (kEnableInferenceDebug &&
      ((current_time - last_raw_debug_time) >= 1000)) {
    TF_LITE_REPORT_ERROR(
        error_reporter, "Raw %s=%d %s=%d %s=%d", kCategoryLabels[0],
        output->data.int8[0] + 128, kCategoryLabels[1],
        output->data.int8[1] + 128, kCategoryLabels[2],
        output->data.int8[2] + 128);
    last_raw_debug_time = current_time;
  }

  // Determine whether a command was recognized based on the output of inference
  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "RecognizeCommands::ProcessLatestResults() failed");
    return;
  }

  static int32_t last_debug_time = 0;
  if (kEnableInferenceDebug &&
      ((current_time - last_debug_time) >= 1000)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Top %s (%d) new=%d @%dms",
                         found_command, score, is_new_command, current_time);
    last_debug_time = current_time;
  }

  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
  RespondToCommand(error_reporter, current_time, found_command, score,
                   is_new_command);
}
