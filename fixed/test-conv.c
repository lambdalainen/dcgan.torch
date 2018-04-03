#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"
#include "col2im.h"

static int dilationW = 1;
static int dilationH = 1;
static int spatial_full_conv_layer;
static int spatial_full_conv_layer_fixed;
static int spatial_batch_norm_layer;
static int spatial_batch_norm_layer_fixed;

static void free_q(struct Q *q)
{
    free(q->f);
    free(q->q);
    free(q->q32);
    free(q);
}

static struct Q *quantize(float *data, long count)
{
    struct Q *q = calloc(1, sizeof(struct Q));
    q->f = data;
    q->q = malloc(sizeof(uint8_t) * count);

    float min = FLT_MAX;
    float max = FLT_MIN;
    double scale;
    double initial_zero_point;
    uint8_t zero_point;

    for (long i = 0; i < count; i++) {
        if (data[i] < min)
            min = data[i];
        else if (data[i] > max)
            max = data[i];
    }
    scale = (max - min) / 255;
    initial_zero_point = - min / scale;
    if (initial_zero_point < 0)
        zero_point = 0;
    else if (initial_zero_point > 255)
        zero_point = 255;
    else
        zero_point = round(initial_zero_point);

    q->min = min;
    q->max = max;
    q->s = scale;
    q->z = zero_point;

    for (long i = 0; i < count; i++) {
        float transformed_val = zero_point + data[i] / scale;
        if (transformed_val < 0.0f)
            transformed_val = 0.0f;
        else if (transformed_val > 255.0f)
            transformed_val = 255.0f;
        q->q[i] = round(transformed_val);
    }
    return q;
}

// Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
// Section 2.4, on bias handling
static struct Q *quantize_bias(float *data, long count, float scale)
{
    struct Q *q = calloc(1, sizeof(struct Q));
    q->f = data;
    q->q32 = malloc(sizeof(int32_t) * count);

    q->s = scale;
    q->z = 0;

    for (long i = 0; i < count; i++) {
        q->q32[i] = round(data[i] / scale);
    }
    return q;
}

static void SpatialFullConvolution(
    float *input,
    float *weight,
    float *bias,
    float *output,
    float *columns,
    long batchSize,
    long nInputPlane,
    long inputWidth,
    long inputHeight,
    long nOutputPlane,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH)
{
  spatial_full_conv_layer++;

  // dilationH and dilationW are constant 1 for transposed convolution
  long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1);
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1);

  // Define a buffer of ones, for bias accumulation
  long ones_n = outputHeight * outputWidth;
  float *ones = calloc(ones_n, sizeof(float));
  for (int i = 0; i < ones_n; i++)
      ones[i] = 1;

  printf("kW %i, kH %i, dW %i, dH %i, padW %i, padH %i, dilationW %i, dilationH %i\n",
          kW, kH, dW, dH, padW, padH, dilationW, dilationH);
  printf("nInputPlane %li, nOutputPlane %li\n", nInputPlane, nOutputPlane);
  printf("inputHeight %li, inputWidth %li, outputHeight %li, outputWidth %li, batchSize %li\n",
          inputHeight, inputWidth, outputHeight, outputWidth, batchSize);

  // Helpers
  float *input_n;
  float *output_n;

  for (int elt = 0; elt < batchSize; elt++) {
    // Matrix mulitply per output:
    // Note: input + 1 means addr + sizeof(float)
    input_n = input + elt * nInputPlane * inputHeight * inputWidth;
    output_n = output + elt * nOutputPlane * outputHeight * outputWidth;

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    // long m = weight->size[1] * weight->size[2] * weight->size[3];
    // long n = columns->size[1];
    // long k = weight->size[0];
    // m and n seem to have been mistakenly swapped in the original code
    long m = inputHeight * inputWidth;
    long n = nOutputPlane * kW * kH;
    long k = nInputPlane;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    gemm(
        'n', 't',
        m, n, k,
        1,
        input_n, m,
        weight, n,
        0,
        columns, m
    );
    //printf("after gemm, output %p, output_n %p\n", output, output_n);

    // Unpack columns back into input:
    col2im(
      columns,
      nOutputPlane, outputHeight, outputWidth, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      dilationH, dilationW,
      output_n
    );

    // Do Bias after:
    long m_ = outputHeight * outputWidth;
    long n_ = nOutputPlane;
    long k_ = 1;

    // alpha * A * B + beta * C (alpha = beta = 1)
    gemm(
        't', 'n',
        m_, n_, k_,
        1,
        ones, k_,
        bias, k_,
        1,
        output_n, m_ // beta == 1: c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
    );
  }

  free(ones);
  printf("### finished: spatial_full_conv_layer %i\n", spatial_full_conv_layer);
}

static void SpatialBatchNormalization(
  float *input,
  float *output,
  float *weight,
  float *bias,
  long batchSize,
  long nInputPlane, // input->size[1]
  long inputWidth,
  long inputHeight)
{
  spatial_batch_norm_layer++;

  double eps = 0.00001;
  long nOutputPlane = nInputPlane;
  long n = batchSize * inputWidth * inputHeight;
  long input_plane_stride = inputWidth * inputHeight;
  long output_plane_stride = input_plane_stride;

  // The input dimensions are: (batchSize, nInputPlane, kW, kH), the output has the same dimensions.
  // Now we are looping through nInputPlane instead of batchSize, therefore we can't simply use
  // a pointer to point to a continuous memory.
  for (long f = 0; f < nInputPlane; ++f) {
    float *in = input + f * input_plane_stride;
    float *out = output + f * output_plane_stride;

    float mean, invstd;

    // compute mean per input
    // torch: if real = float, accreal = double
    double sum = 0;
    for (int i = 0; i < batchSize; i++) {
        float *plane_ptr = in + i * nInputPlane * input_plane_stride;
        for (float *p = plane_ptr; p < (plane_ptr + input_plane_stride); p++)
            sum += *p;
    }

    mean = (float) sum / n;

    // compute variance per input
    sum = 0;
    for (int i = 0; i < batchSize; i++) {
        float *plane_ptr = in + i * nInputPlane * input_plane_stride;
        for (float *p = plane_ptr; p < (plane_ptr + input_plane_stride); p++)
            sum += (*p - mean) * (*p - mean);
    }

    invstd = (float) (1 / sqrt(sum/n + eps));

    // compute output
    float w = *(weight + f);
    float b = *(bias + f);

    // write output
    for (int i = 0; i < batchSize; i++) {
        float *input_plane_ptr = in + i * nInputPlane * input_plane_stride;
        float *output_plane_ptr = out + i * nOutputPlane * output_plane_stride;
        float *p, *q;
        for (p = input_plane_ptr, q = output_plane_ptr;
             p < (input_plane_ptr + input_plane_stride) && q < (output_plane_ptr + output_plane_stride);
             p++, q++) {
            *q = (float) (((*p - mean) * invstd) * w + b);
        }
    }
  }
  printf("### finished: spatial_batch_norm_layer %i\n", spatial_batch_norm_layer);
}

static void SpatialFullConvolution_fixed(
    struct Q *input_q,
    struct Q *weight_q,
    struct Q *bias_q,
    struct Q *output_q,
    int32_t *output,
    int32_t *columns,
    long batchSize,
    long nInputPlane,
    long inputWidth,
    long inputHeight,
    long nOutputPlane,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH)
{
  spatial_full_conv_layer_fixed++;

  // dilationH and dilationW are constant 1 for transposed convolution
  long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1);
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1);

  printf("kW %i, kH %i, dW %i, dH %i, padW %i, padH %i, dilationW %i, dilationH %i\n",
          kW, kH, dW, dH, padW, padH, dilationW, dilationH);
  printf("nInputPlane %li, nOutputPlane %li\n", nInputPlane, nOutputPlane);
  printf("inputHeight %li, inputWidth %li, outputHeight %li, outputWidth %li, batchSize %li\n",
          inputHeight, inputWidth, outputHeight, outputWidth, batchSize);

  // Helpers
  uint8_t *input_n;
  int32_t *output_n;

  for (int elt = 0; elt < batchSize; elt++) {
    // Matrix mulitply per output:
    // Note: input + 1 means addr + sizeof(uint8_t)
    input_n = input_q->q + elt * nInputPlane * inputHeight * inputWidth;
    output_n = output + elt * nOutputPlane * outputHeight * outputWidth;

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    // long m = weight->size[1] * weight->size[2] * weight->size[3];
    // long n = columns->size[1];
    // long k = weight->size[0];
    // m and n seem to have been mistakenly swapped in the original code
    long m = inputHeight * inputWidth;
    long n = nOutputPlane * kW * kH;
    long k = nInputPlane;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    gemm_fixed(
        'n', 't',
        m, n, k,
        1,
        input_n, m,
        weight_q->q, n,
        0,
        columns, m,
        input_q, weight_q, output_q
    );

    // Unpack columns back into input:
    col2im_fixed(
      columns,
      nOutputPlane, outputHeight, outputWidth, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      dilationH, dilationW,
      output_n
    );

    // Add bias and scale down to uint8_t
    long output_plane_size = outputWidth * outputHeight;
    float scale = input_q->s * weight_q->s / output_q->s;

    for (long j = 0; j < nOutputPlane; j++) {
        int32_t b = bias_q->q32[j];
        for (long k = 0; k < output_plane_size; k++) {
            long idx = j*output_plane_size + k;
            int32_t output = output_n[idx] + b;
            int32_t result = round(output * scale) + output_q->z;
            if (result < 0)
                result = 0;
            else if (result > 255)
                result = 255;
            output_n[idx] = result;
        }
    }
  }

  printf("### finished: spatial_full_conv_layer_fixed %i\n", spatial_full_conv_layer_fixed);
}

static void SpatialBatchNormalization_fixed(
  int32_t *input,
  int32_t *output,
  uint8_t *weight,
  int32_t *bias,
  long batchSize,
  long nInputPlane, // input->size[1]
  long inputWidth,
  long inputHeight)
{
  spatial_batch_norm_layer_fixed++;

  double eps = 0.00001;
  long nOutputPlane = nInputPlane;
  long n = batchSize * inputWidth * inputHeight;
  long input_plane_stride = inputWidth * inputHeight;
  long output_plane_stride = input_plane_stride;

  // The input dimensions are: (batchSize, nInputPlane, kW, kH), the output has the same dimensions.
  // Now we are looping through nInputPlane instead of batchSize, therefore we can't simply use
  // a pointer to point to a continuous memory.
  for (long f = 0; f < nInputPlane; ++f) {
    int32_t *in = input + f * input_plane_stride;
    int32_t *out = output + f * output_plane_stride;

    int32_t mean, invstd;

    // compute mean per input
    // torch: if real = float, accreal = double
    int32_t sum = 0;
    for (int i = 0; i < batchSize; i++) {
        int32_t *plane_ptr = in + i * nInputPlane * input_plane_stride;
        for (int32_t *p = plane_ptr; p < (plane_ptr + input_plane_stride); p++)
            sum += *p;
    }

    mean = (int32_t) sum / n;

    // compute variance per input
    sum = 0;
    for (int i = 0; i < batchSize; i++) {
        int32_t *plane_ptr = in + i * nInputPlane * input_plane_stride;
        for (int32_t *p = plane_ptr; p < (plane_ptr + input_plane_stride); p++)
            sum += (*p - mean) * (*p - mean);
    }

    invstd = (int32_t) (1 / sqrt(sum/n + eps));

    // compute output
    int32_t w = *(weight + f);
    int32_t b = *(bias + f);

    // write output
    for (int i = 0; i < batchSize; i++) {
        int32_t *input_plane_ptr = in + i * nInputPlane * input_plane_stride;
        int32_t *output_plane_ptr = out + i * nOutputPlane * output_plane_stride;
        int32_t *p, *q;
        for (p = input_plane_ptr, q = output_plane_ptr;
             p < (input_plane_ptr + input_plane_stride) && q < (output_plane_ptr + output_plane_stride);
             p++, q++) {
            *q = (int32_t) (((*p - mean) * invstd) * w + b);
        }
    }
  }
  printf("### finished: spatial_batch_norm_layer_fixed %i\n", spatial_batch_norm_layer_fixed);
}


static struct Q *forward_SpatialFullConvolution(
    int layer,
    struct Q *input_q,
    long batchSize,
    long nInputPlane,
    long inputWidth,
    long inputHeight,
    long nOutputPlane,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH)
{
    char weight_path[256];
    char bias_path[256];
    char output_path[256];
    sprintf(weight_path, "../bin/weight_%i.bin", layer);
    sprintf(bias_path, "../bin/bias_%i.bin", layer);
    sprintf(output_path, "../bin/output_%i_test.bin", layer);

    long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1);
    long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1);
    long output_size = batchSize * nOutputPlane * outputWidth * outputHeight;
    float *output = calloc(output_size, sizeof(float));
    long weight_size = nInputPlane * nOutputPlane * kW * kH;
    float *weight = calloc(weight_size, sizeof(float));
    float *bias = calloc(nOutputPlane, sizeof(float));
    // columns: (nOutputPlane*kW*kH, inputHeight*inputWidth)
    float *columns = calloc(nOutputPlane * kW * kH * inputHeight * inputWidth, sizeof(float));

    FILE *fp = fopen(weight_path, "rb");
    fread(weight, sizeof(float), weight_size, fp);
    fclose(fp);
    fp = fopen(bias_path, "rb");
    fread(bias, sizeof(float), nOutputPlane, fp);
    fclose(fp);

    SpatialFullConvolution(
        input_q->f, weight, bias, output, columns,
        batchSize, nInputPlane, inputWidth, inputHeight, nOutputPlane,
        kW, kH, dW, dH, padW, padH);

    fp = fopen(output_path, "wb");
    fwrite(output, sizeof(float), output_size, fp);
    fclose(fp);

    // ----------

    struct Q *weight_q = quantize(weight, weight_size);
    struct Q *bias_q = quantize_bias(bias, nOutputPlane, input_q->s * weight_q->s);
    struct Q *output_q = quantize(output, output_size);

    printf("weight_q: min %f max %f scale %f zero_point %i\n", weight_q->min, weight_q->max, weight_q->s, weight_q->z);
    printf("bias_q: scale %f zero_point %i\n", bias_q->s, bias_q->z);
    printf("output_q: min %f max %f scale %f zero_point %i\n", output_q->min, output_q->max, output_q->s, output_q->z);

    int32_t *output_fixed = calloc(output_size, sizeof(int32_t));
    int32_t *columns_fixed = calloc(nOutputPlane * kW * kH * inputHeight * inputWidth, sizeof(int32_t));

    SpatialFullConvolution_fixed(
        input_q, weight_q, bias_q, output_q, output_fixed, columns_fixed,
        batchSize, nInputPlane, inputWidth, inputHeight, nOutputPlane,
        kW, kH, dW, dH, padW, padH);

    // output dequantized
    float *output_deq = calloc(output_size, sizeof(float));
    for (int i = 0; i < output_size; i++) {
        output_deq[i] = output_q->s * (output_fixed[i] - output_q->z);
    }

    // At this point we have 2 groups of data we can compare:
    // output vs output_deq, output_fixed vs output_q->q
    int diff_sum_fixed = 0;
    int diff_abs_sum_fixed = 0;
    int diff_squared_sum_fixed = 0;
    for (int i = 0; i < output_size; i++) {
        int diff = output_fixed[i] - output_q->q[i];
        diff_sum_fixed += diff;
        diff_abs_sum_fixed += abs(diff);
        diff_squared_sum_fixed += diff * diff;
    }
    printf("--- fixed diffs:\n");
    printf("sum diff: %i average diff: %f\n", diff_sum_fixed, (float)diff_sum_fixed / output_size);
    printf("average absolute diff: %f, RMS diff: %f\n", (float)diff_abs_sum_fixed / output_size,
            sqrtf((float)diff_squared_sum_fixed / output_size));

    float diff_sum = 0.0f;
    float diff_abs_sum = 0.0f;
    float diff_squared_sum = 0.0f;
    for (int i = 0; i < output_size; i++) {
        float diff = output_deq[i] - output[i];
        diff_sum += diff;
        diff_abs_sum += fabsf(diff);
        diff_squared_sum += diff * diff;
    }
    printf("--- float diffs:\n");
    printf("sum diff: %f average diff: %f\n", diff_sum, diff_sum / output_size);
    printf("average absolute diff: %f, RMS diff: %f\n", diff_abs_sum / output_size,
            sqrtf(diff_squared_sum / output_size));

    free_q(input_q);
    free_q(weight_q);
    free_q(bias_q);
    free(columns);
    free(columns_fixed);

    output_q->q32 = output_fixed;
    return output_q;
}

static struct Q *forward_SpatialBatchNormalization(
    int layer,
    struct Q *input_q,
    long batchSize,
    long nInputPlane,
    long inputWidth,
    long inputHeight)
{
    printf("\n");
    char weight_path[256];
    char bias_path[256];
    char output_path[256];
    sprintf(weight_path, "../bin/weight_%i.bin", layer);
    sprintf(bias_path, "../bin/bias_%i.bin", layer);
    sprintf(output_path, "../bin/output_%i_test.bin", layer);

    // (64, 512, 4, 4)
    // same shape for input and output
    long output_size = batchSize * nInputPlane * inputWidth * inputHeight;
    float *output = calloc(output_size, sizeof(float));
    // (512)
    float *weight = calloc(nInputPlane, sizeof(float));
    // (512)
    float *bias = calloc(nInputPlane, sizeof(float));

    FILE *fp = fopen(weight_path, "rb");
    fread(weight, sizeof(float), nInputPlane, fp);
    fclose(fp);
    fp = fopen(bias_path, "rb");
    fread(bias, sizeof(float), nInputPlane, fp);
    fclose(fp);

    SpatialBatchNormalization(
        input_q->f, output, weight, bias,
        batchSize, nInputPlane, inputWidth, inputHeight);

    fp = fopen(output_path, "wb");
    fwrite(output, sizeof(float), output_size, fp);
    fclose(fp);

    // ----------

    struct Q *weight_q = quantize(weight, nInputPlane);
    struct Q *bias_q = quantize_bias(bias, nInputPlane, input_q->s * weight_q->s);
    struct Q *output_q = quantize(output, output_size);

    printf("weight_q: min %f max %f scale %f zero_point %i\n", weight_q->min, weight_q->max, weight_q->s, weight_q->z);
    printf("bias_q: scale %f zero_point %i\n", bias_q->s, bias_q->z);
    printf("output_q: min %f max %f scale %f zero_point %i\n", output_q->min, output_q->max, output_q->s, output_q->z);

    int32_t *output_fixed = calloc(output_size, sizeof(int32_t));

    SpatialBatchNormalization_fixed(
        input_q->q32, output_fixed, weight_q->q, bias_q->q32,
        batchSize, nInputPlane, inputWidth, inputHeight);

    free_q(input_q);
    free_q(weight_q);
    free_q(bias_q);

    output_q->q32 = output_fixed;
    return output_q;
}

int main(void)
{
    float *input_1f = calloc(64 * 100, sizeof(float));
    FILE *fp = fopen("../bin/input_1.bin", "rb");
    fread(input_1f, sizeof(float), 64 * 100, fp);
    fclose(fp);

    // quantize input_1f
    struct Q *input_1q = quantize(input_1f, 64 * 100);
    printf("input_1q: min %f max %f scale %f zero_point %i\n", input_1q->min, input_1q->max, input_1q->s, input_1q->z);

    // (64, 100, 1, 1) -> (64, 512, 4, 4)
    struct Q *output_1q = forward_SpatialFullConvolution(
        1, input_1q, 64, 100, 1, 1, 512, 4, 4, 1, 1, 0, 0);

    // output_1q->q is the quantized values from output_1q->f
    // output_1q->q32 is the actual output of SpatialFullConvolution_fixed
    //
    // (64, 512, 4, 4) -> (64, 512, 4, 4)
    struct Q *output_2q = forward_SpatialBatchNormalization(
        2, output_1q, 64, 512, 4, 4);

    free_q(output_2q);

    return 0;
}
