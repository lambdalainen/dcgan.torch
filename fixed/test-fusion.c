#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"
#include "col2im.h"

#define read_bin(type, path, data, count) do { \
    FILE *fp = fopen(path, "rb"); \
    fread(data, sizeof(type), count, fp); \
    fclose(fp); \
} while (0)

#define save_bin(type, path, data, count) do { \
    FILE *fp = fopen(path, "wb"); \
    fwrite(data, sizeof(type), count, fp); \
    fclose(fp); \
} while (0)

#define save_txt(path, data, count) do { \
    FILE *fp = fopen(path, "wb"); \
    for (long i = 0; i < count; i++) \
        fprintf(fp, "%x\n", data[i]); \
    fclose(fp); \
} while (0)

static int dilationW = 1;
static int dilationH = 1;
static int spatial_full_conv_layer;
static int spatial_full_conv_layer_fixed;
static int spatial_batch_norm_layer;
static int relu_layer;
static int tanh_layer;

static void free_q(struct Q *q)
{
    free(q->f);
    free(q->f2);
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
static struct Q *quantize_int32(float *data, long count, float scale)
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

static void compare_output_float(float *output, float *output_2, int output_count)
{
    float diff_sum = 0.0f;
    float diff_abs_sum = 0.0f;
    float diff_squared_sum = 0.0f;
    for (int i = 0; i < output_count; i++) {
        float diff = output_2[i] - output[i];
        diff_sum += diff;
        diff_abs_sum += fabsf(diff);
        diff_squared_sum += diff * diff;
    }
    printf("--- float diffs:\n");
    printf("sum diff: %f average diff: %f\n", diff_sum, diff_sum / output_count);
    printf("average absolute diff: %f, RMS diff: %f\n", diff_abs_sum / output_count,
            sqrtf(diff_squared_sum / output_count));
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

#if 1
    // gemm & col2im combined
    sparse_gemm(
        m, n, k,
        input_n, m,
        weight, n,
        nOutputPlane,
        outputHeight, outputWidth,
        inputHeight, inputWidth,
        kH, kW, padH, padW, dH, dW,
        dilationH, dilationW,
        output_n
    );
#else
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
#endif

    // Do Bias after:
#if 1
    long output_plane_size = outputWidth * outputHeight;

    for (long j = 0; j < nOutputPlane; j++) {
        float b = bias[j];
        for (long k = 0; k < output_plane_size; k++) {
            output_n[j*output_plane_size + k] += b;
        }
    }
#else
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
#endif
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
  long inputHeight,
  int increase_layer)
{
  if (increase_layer)
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
    if (f == 0)
        printf("%s: <sum %f mean %f>\n", __func__, sum, mean);

    // compute variance per input
    sum = 0;
    for (int i = 0; i < batchSize; i++) {
        float *plane_ptr = in + i * nInputPlane * input_plane_stride;
        for (float *p = plane_ptr; p < (plane_ptr + input_plane_stride); p++)
            sum += (*p - mean) * (*p - mean);
    }

    invstd = (float) (1 / sqrt(sum/n + eps));
    if (f == 0)
        printf("%s: <sum %f invstd %f>\n", __func__, sum, invstd);

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
#if 0
            if (f == 0 && i == 0)
                printf("%f\n", (*p - mean) * invstd);
#endif
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

#if 1
    // gemm & col2im combined
    sparse_gemm_fixed(
        m, n, k,
        input_n, m,
        weight_q->q, n,
        nOutputPlane,
        outputHeight, outputWidth,
        inputHeight, inputWidth,
        kH, kW, padH, padW, dH, dW,
        dilationH, dilationW,
        output_n,
        input_q, weight_q, output_q,
        elt == 0 ? spatial_full_conv_layer_fixed : 0
    );
#else
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
#endif

    // Add bias and scale down to uint8_t
    long output_plane_area = outputWidth * outputHeight;
    //float scale = input_q->s * weight_q->s / output_q->s;

    //int ii = 0;
    for (long j = 0; j < nOutputPlane; j++) {
        int32_t b = bias_q->q32[j];
        for (long k = 0; k < output_plane_area; k++) {
            long idx = j*output_plane_area + k;
            int32_t output = output_n[idx] + b;
#if 0
            if (elt == 0) {
                printf("%04i: %08x \t + %i -> %08x\n", ii, output_n[idx], b, output);
                ii++;
            }
#endif

#if 0 // to scale down to 8-bit or not
            int32_t result = round(output * scale) + output_q->z;
            if (result < 0)
                result = 0;
            else if (result > 255)
                result = 255;
            output_n[idx] = result;
#else
            output_n[idx] = output;
#endif
        }
    }
  }

  printf("### finished: spatial_full_conv_layer_fixed %i\n", spatial_full_conv_layer_fixed);
}

static void ReLU(
    float *input,
    float *output,
    long count,
    int increase_layer)
{
    if (increase_layer)
        relu_layer++;

    float *p, *q;
    for (p = input, q = output;
         p < (input + count) && q < (output + count);
         p++, q++) {
        *q = *p > 0 ? *p : 0;
    }
    printf("### finished: relu_layer %i\n", relu_layer);
}

static void Tanh(
  float *input,
  float *output,
  long size)
{
    tanh_layer++;

    float *p, *q;
    for (p = input, q = output;
         p < (input + size) && q < (output + size);
         p++, q++) {
        *q = tanhf(*p);
    }
    printf("### finished: tanh_layer %i\n", tanh_layer);
}

#if 0
static void Tanh_fixed(
  struct Q *input_q,
  uint8_t *output,
  long size)
{
    tanh_layer_fixed++;

    int32_t *p;
    uint8_t *q;
    for (p = input_q->q32, q = output;
         p < (input_q->q32 + size) && q < (output + size);
         p++, q++) {
        *q = tanhf(*p);
    }
    printf("### finished: tanh_layer_fixed %i\n", tanh_layer_fixed);
}
#endif

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
    int padH,
    float *scale_axb)
{
    printf("\n===>>> %i\n", layer);
    char path[256];
    long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1);
    long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1);
    long output_count = batchSize * nOutputPlane * outputWidth * outputHeight;
    float *output = calloc(output_count, sizeof(float));
    long weight_count = nInputPlane * nOutputPlane * kW * kH;
    float *weight = calloc(weight_count, sizeof(float));
    float *bias = calloc(nOutputPlane, sizeof(float));
    // columns: (nOutputPlane*kW*kH, inputHeight*inputWidth)
    float *columns = calloc(nOutputPlane * kW * kH * inputHeight * inputWidth, sizeof(float));

    sprintf(path, "../bin/weight_%i.bin", layer);
    read_bin(float, path, weight, weight_count);
    sprintf(path, "../bin/bias_%i.bin", layer);
    read_bin(float, path, bias, nOutputPlane);

    SpatialFullConvolution(
        input_q->f, weight, bias, output, columns,
        batchSize, nInputPlane, inputWidth, inputHeight, nOutputPlane,
        kW, kH, dW, dH, padW, padH);

    sprintf(path, "../bin/output_%i_test.bin", layer);
    save_bin(float, path, output, output_count);

    // ----------

    struct Q *weight_q = quantize(weight, weight_count);
    *scale_axb = input_q->s * weight_q->s;
    struct Q *bias_q = quantize_int32(bias, nOutputPlane, *scale_axb);
    struct Q *output_q = quantize(output, output_count);

    if (layer == 1) {
        // save to .mem file, used to initialize BRAM
        sprintf(path, "../bin/input_%i_uint8.mem", layer);
        save_txt(path, input_q->q, nInputPlane); // only save 1 batch
    }

    // transpose weight_q and save to .bin file, used to generate .mcs file to program SPI Flash
    uint8_t *weight_fpga = calloc(weight_count, sizeof(uint8_t));
    int n = nOutputPlane * kW * kH;
    int k = nInputPlane;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            weight_fpga[j * k + i] = weight_q->q[i * n + j];
    sprintf(path, "../bin/weight_%i_uint8.bin", layer);
    save_bin(uint8_t, path, weight_fpga, weight_count);
    sprintf(path, "../bin/weight_%i_uint8.mem", layer);
    save_txt(path, weight_fpga, weight_count);
    sprintf(path, "../bin/bias_%i_int32.mem", layer);
    save_txt(path, bias_q->q32, nOutputPlane);

    printf("input_q: min %f max %f scale %f zero_point %i\n", input_q->min, input_q->max, input_q->s, input_q->z);
    printf("weight_q: min %f max %f scale %f zero_point %i\n", weight_q->min, weight_q->max, weight_q->s, weight_q->z);
    printf("bias_q: scale %f zero_point %i\n", bias_q->s, bias_q->z);
    printf("output_q: min %f max %f scale %f zero_point %i\n", output_q->min, output_q->max, output_q->s, output_q->z);
    printf("term_4: %i\n", input_q->z * weight_q->z * k);

    int32_t *output_fixed = calloc(output_count, sizeof(int32_t));
    int32_t *columns_fixed = calloc(nOutputPlane * kW * kH * inputHeight * inputWidth, sizeof(int32_t));

    SpatialFullConvolution_fixed(
        input_q, weight_q, bias_q, output_q, output_fixed, columns_fixed,
        batchSize, nInputPlane, inputWidth, inputHeight, nOutputPlane,
        kW, kH, dW, dH, padW, padH);

#if 0 // only for when output of transconv is scaled down to uint8_t
    // output dequantized
    float *output_deq = calloc(output_count, sizeof(float));
    for (int i = 0; i < output_count; i++) {
        output_deq[i] = output_q->s * (output_fixed[i] - output_q->z);
    }

    printf("First 10 output values (float vs dequantized):\n");
    for (int i = 0; i < 10; i++)
        printf("%f ", output[i]);
    printf("\n");
    for (int i = 0; i < 10; i++)
        printf("%f ", output_deq[i]);
    printf("\n");

    // At this point we have 2 groups of data we can compare:
    // output vs output_deq: original float output vs dequantized float from uint8_t
    // output_fixed vs output_q->q: fixed uint8_t output vs quantized uint8_t from original float output
    int diff_sum_fixed = 0;
    int diff_abs_sum_fixed = 0;
    int diff_squared_sum_fixed = 0;
    for (int i = 0; i < output_count; i++) {
        int diff = output_fixed[i] - output_q->q[i];
        diff_sum_fixed += diff;
        diff_abs_sum_fixed += abs(diff);
        diff_squared_sum_fixed += diff * diff;
    }
    printf("--- fixed diffs:\n");
    printf("sum diff: %i average diff: %f\n", diff_sum_fixed, (float)diff_sum_fixed / output_count);
    printf("average absolute diff: %f, RMS diff: %f\n", (float)diff_abs_sum_fixed / output_count,
            sqrtf((float)diff_squared_sum_fixed / output_count));

    compare_output_float(output, output_deq, output_count);
    free(output_deq);
#endif

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
    long inputHeight,
    float scale_axb)
{
    printf("\n");
    char path[256];

    // (64, 512, 4, 4)
    // same shape for input and output
    long output_count = batchSize * nInputPlane * inputWidth * inputHeight;
    float *output = calloc(output_count, sizeof(float));
    // (512)
    float *weight = calloc(nInputPlane, sizeof(float));
    // (512)
    float *bias = calloc(nInputPlane, sizeof(float));

    sprintf(path, "../bin/weight_%i.bin", layer);
    read_bin(float, path, weight, nInputPlane);
    sprintf(path, "../bin/bias_%i.bin", layer);
    read_bin(float, path, bias, nInputPlane);

    SpatialBatchNormalization(
        input_q->f, output, weight, bias,
        batchSize, nInputPlane, inputWidth, inputHeight, 1);

    sprintf(path, "../bin/output_%i_test.bin", layer);
    save_bin(float, path, output, output_count);

    // ----------

    float *input_deq = calloc(output_count, sizeof(float));
    for (int i = 0; i < output_count; i++) {
        input_deq[i] = scale_axb * input_q->q32[i];
    }
    printf("First 10 inputs to BN layer (float vs dequantized):\n");
    for (int i = 0; i < 10; i++)
        printf("%f ", input_q->f[i]);
    printf("\n");
    for (int i = 0; i < 10; i++)
        printf("%f ", input_deq[i]);
    printf("\n");

    /*
    sprintf(path, "../bin/weight_%i_int32.mem", layer);
    save_txt(path, weight_q->q32, nInputPlane);
    sprintf(path, "../bin/bias_%i_int32.mem", layer);
    save_txt(path, bias_q->q32, nInputPlane);
    */

    struct Q *output_q = quantize(output, output_count);

    float *output2 = calloc(output_count, sizeof(float));
    SpatialBatchNormalization(
        input_deq, output2, weight, bias,
        batchSize, nInputPlane, inputWidth, inputHeight, 0);

    printf("First 10 output values (output vs output2):\n");
    for (int i = 0; i < 10; i++)
        printf("%f ", output[i]);
    printf("\n");
    for (int i = 0; i < 10; i++)
        printf("%f ", output2[i]);
    printf("\n");

    compare_output_float(output, output2, output_count);

    free_q(input_q);
    free(input_deq);
    free(weight);
    free(bias);

    output_q->f2 = output2;
    return output_q;
}

static struct Q *forward_ReLU(
    int layer,
    struct Q *input_q,
    long batchSize,
    long nInputPlane,
    long inputWidth,
    long inputHeight)
{
    printf("\n");
    char path[256];

    long output_count = batchSize * nInputPlane * inputWidth * inputHeight;
    float *output = calloc(output_count, sizeof(float));

    ReLU(input_q->f, output, output_count, 1);

    sprintf(path, "../bin/output_%i_test.bin", layer);
    save_bin(float, path, output, output_count);

    // ----------

    float *output2 = calloc(output_count, sizeof(float));
    ReLU(input_q->f2, output2, output_count, 0);

    printf("Non-zero values from the first 100 outputs (output vs output2):\n");
    for (int i = 0; i < 100; i++)
        if (output[i] != 0.0f)
            printf("%f ", output[i]);
    printf("\n");
    for (int i = 0; i < 100; i++)
        if (output2[i] != 0.0f)
            printf("%f ", output2[i]);
    printf("\n");

    compare_output_float(output, output2, output_count);

    // Since it's just regular floating-point computation for BN and ReLU, here we simply need to
    // quantize the float data itself. It is possible, of course, to have a predetermined s and z
    // so we don't have to compute them on FPGA, but let's see how it goes.
    struct Q *output_q = quantize(output2, output_count);
    printf("output_q: min %f max %f scale %f zero_point %i\n", output_q->min, output_q->max, output_q->s, output_q->z);

    free_q(input_q);
    free(output2);
    output_q->f = output; // pass the original f down
    return output_q;
}

static struct Q *forward_Tanh(
    int layer,
    struct Q *input_q,
    long batchSize,
    long nInputPlane,
    long inputWidth,
    long inputHeight,
    float scale_axb)
{
    printf("\n");
    char path[256];

    long output_count = batchSize * nInputPlane * inputWidth * inputHeight;
    float *output = calloc(output_count, sizeof(float));

    float *input_deq = calloc(output_count, sizeof(float));
    for (int i = 0; i < output_count; i++) {
        input_deq[i] = scale_axb * input_q->q32[i];
    }

    printf("First 10 inputs to Tanh layer (float vs dequantized):\n");
    for (int i = 0; i < 10; i++)
        printf("%f ", input_q->f[i]);
    printf("\n");
    for (int i = 0; i < 10; i++)
        printf("%f ", input_deq[i]);
    printf("\n");

    Tanh(input_deq, output, output_count);

    sprintf(path, "../bin/output_%i_test.bin", layer);
    save_bin(float, path, output, output_count);

    // ----------

    struct Q *output_q = quantize(output, output_count);
    printf("output_q: min %f max %f scale %f zero_point %i\n", output_q->min, output_q->max, output_q->s, output_q->z);

    free(input_deq);
    free_q(input_q);
    return output_q;
}

int main(void)
{
    float scale_axb = 0.0f; // S_A * S_B

    float *input_1f = calloc(64 * 100, sizeof(float));
    read_bin(float, "../bin/input_1.bin", input_1f, 64 * 100);

    // quantize input_1f
    struct Q *input_1q = quantize(input_1f, 64 * 100);
    printf("input_1q: min %f max %f scale %f zero_point %i\n", input_1q->min, input_1q->max, input_1q->s, input_1q->z);

    struct Q *output_1q = forward_SpatialFullConvolution(
        1, input_1q, 64, 100, 1, 1, 512, 4, 4, 1, 1, 0, 0, &scale_axb);

    // output_1q->q is the quantized values from output_1q->f
    // output_1q->q32 is the actual output of SpatialFullConvolution_fixed
    struct Q *output_2q = forward_SpatialBatchNormalization(
        2, output_1q, 64, 512, 4, 4, scale_axb);

    struct Q *output_3q = forward_ReLU(3, output_2q, 64, 512, 4, 4);

    struct Q *output_4q = forward_SpatialFullConvolution(
        4, output_3q, 64, 512, 4, 4, 256, 4, 4, 2, 2, 1, 1, &scale_axb);

    struct Q *output_5q = forward_SpatialBatchNormalization(
        5, output_4q, 64, 256, 8, 8, scale_axb);

    struct Q *output_6q = forward_ReLU(6, output_5q, 64, 256, 8, 8);

    struct Q *output_7q = forward_SpatialFullConvolution(
        7, output_6q, 64, 256, 8, 8, 128, 4, 4, 2, 2, 1, 1, &scale_axb);

    struct Q *output_8q = forward_SpatialBatchNormalization(
        8, output_7q, 64, 128, 16, 16, scale_axb);

    struct Q *output_9q = forward_ReLU(9, output_8q, 64, 128, 16, 16);

    struct Q *output_10q = forward_SpatialFullConvolution(
        10, output_9q, 64, 128, 16, 16, 64, 4, 4, 2, 2, 1, 1, &scale_axb);

    struct Q *output_11q = forward_SpatialBatchNormalization(
        11, output_10q, 64, 64, 32, 32, scale_axb);

    struct Q *output_12q = forward_ReLU(12, output_11q, 64, 64, 32, 32);

    struct Q *output_13q = forward_SpatialFullConvolution(
        13, output_12q, 64, 64, 32, 32, 3, 4, 4, 2, 2, 1, 1, &scale_axb);

    struct Q *output_14q = forward_Tanh(14, output_13q, 64, 3, 64, 64, scale_axb);

    free_q(output_14q);

    return 0;
}
