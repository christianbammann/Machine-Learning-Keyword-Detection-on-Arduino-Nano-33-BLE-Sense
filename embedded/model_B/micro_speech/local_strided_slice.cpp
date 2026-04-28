#include "local_strided_slice.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace {

constexpr int kInputTensor = 0;
constexpr int kBeginTensor = 1;
constexpr int kEndTensor = 2;
constexpr int kStridesTensor = 3;
constexpr int kOutputTensor = 0;

int ElementCount(const TfLiteIntArray* dims) {
  if (dims->size == 0) {
    return 1;
  }
  int count = 1;
  for (int i = 0; i < dims->size; ++i) {
    count *= dims->data[i];
  }
  return count;
}

int NormalizeIndex(int index, int size) {
  if (index < 0) {
    index += size;
  }
  if (index < 0) {
    index = 0;
  }
  if (index > size) {
    index = size;
  }
  return index;
}

template <typename InputT, typename OutputT>
TfLiteStatus Slice1D(TfLiteContext* context, TfLiteNode* node,
                     const TfLiteStridedSliceParams* params) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* begin =
      tflite::micro::GetEvalInput(context, node, kBeginTensor);
  const TfLiteEvalTensor* end =
      tflite::micro::GetEvalInput(context, node, kEndTensor);
  const TfLiteEvalTensor* strides =
      tflite::micro::GetEvalInput(context, node, kStridesTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const int input_count = ElementCount(input->dims);
  const int32_t* begin_data = tflite::micro::GetTensorData<int32_t>(begin);
  const int32_t* end_data = tflite::micro::GetTensorData<int32_t>(end);
  const int32_t* strides_data = tflite::micro::GetTensorData<int32_t>(strides);
  const InputT* input_data = tflite::micro::GetTensorData<InputT>(input);
  OutputT* output_data = tflite::micro::GetTensorData<OutputT>(output);

  const int stride = strides_data[0];
  if (stride != 1) {
    return kTfLiteError;
  }

  int start = (params->begin_mask & 1) ? 0 : begin_data[0];
  int stop = (params->end_mask & 1) ? input_count : end_data[0];
  start = NormalizeIndex(start, input_count);
  stop = NormalizeIndex(stop, input_count);

  if (params->shrink_axis_mask & 1) {
    output_data[0] = static_cast<OutputT>(input_data[start]);
    return kTfLiteOk;
  }

  int output_index = 0;
  for (int input_index = start; input_index < stop; ++input_index) {
    output_data[output_index++] = static_cast<OutputT>(input_data[input_index]);
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteStridedSliceParams* params =
      reinterpret_cast<const TfLiteStridedSliceParams*>(node->builtin_data);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  if (input->type == kTfLiteInt32 && output->type == kTfLiteInt32) {
    return Slice1D<int32_t, int32_t>(context, node, params);
  }
  if (input->type == kTfLiteInt32 && output->type == kTfLiteInt64) {
    return Slice1D<int32_t, int64_t>(context, node, params);
  }
  if (input->type == kTfLiteInt64 && output->type == kTfLiteInt64) {
    return Slice1D<int64_t, int64_t>(context, node, params);
  }

  return kTfLiteError;
}

}  // namespace

TfLiteRegistration RegisterLocalStridedSlice() {
  TfLiteRegistration registration = {};
  registration.init = nullptr;
  registration.free = nullptr;
  registration.prepare = nullptr;
  registration.invoke = Eval;
  registration.profiling_string = nullptr;
  registration.builtin_code = 0;
  registration.custom_name = nullptr;
  registration.version = 1;
  return registration;
}
