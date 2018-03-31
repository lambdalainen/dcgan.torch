#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"
#include "col2im.h"

static int dilationW = 1;
static int dilationH = 1;
static int spatial_full_conv_layer;
static int spatial_batch_norm_layer;
static int relu_layer;
static int tanh_layer;

static void SpatialFullConvolution(
    float *input,
    float *output,
    float *weight,
    float *bias,
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
#if 1
    long output_plane_size = outputWidth * outputHeight;

    for (long j = 0; j < nOutputPlane; j++) {
        float b = bias[j];
        for (long k = 0; k < output_plane_size; k++) {
            output_n[j*output_plane_size + k] += b;
        }
    }
#endif
#if 0
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

static void ReLU(
    float *input,
    float *output,
    long size)
{
    relu_layer++;

    float *p, *q;
    for (p = input, q = output;
         p < (input + size) && q < (output + size);
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

static float *forward_SpatialFullConvolution(
    int layer,
    float *input,
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
    sprintf(weight_path, "bin/weight_%i.bin", layer);
    sprintf(bias_path, "bin/bias_%i.bin", layer);
    sprintf(output_path, "bin/output_%i_test.bin", layer);

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
        input, output, weight, bias, columns,
        batchSize, nInputPlane, inputWidth, inputHeight, nOutputPlane,
        kW, kH, dW, dH, padW, padH);

    fp = fopen(output_path, "wb");
    fwrite(output, sizeof(float), output_size, fp);
    fclose(fp);

    free(input);
    free(weight);
    free(bias);
    free(columns);

    return output;
}

static float *forward_SpatialBatchNormalization(
    int layer,
    float *input,
    long batchSize,
    long nInputPlane,
    long inputWidth,
    long inputHeight)
{
    char weight_path[256];
    char bias_path[256];
    char output_path[256];
    sprintf(weight_path, "bin/weight_%i.bin", layer);
    sprintf(bias_path, "bin/bias_%i.bin", layer);
    sprintf(output_path, "bin/output_%i_test.bin", layer);

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
        input, output, weight, bias,
        batchSize, nInputPlane, inputWidth, inputHeight);

    fp = fopen(output_path, "wb");
    fwrite(output, sizeof(float), output_size, fp);
    fclose(fp);

    free(input);
    free(weight);
    free(bias);
    return output;
}

static float *forward_ReLU(
    int layer,
    float *input,
    long batchSize,
    long nInputPlane,
    long inputWidth,
    long inputHeight)
{
    char output_path[256];
    sprintf(output_path, "bin/output_%i_test.bin", layer);

    long output_size = batchSize * nInputPlane * inputWidth * inputHeight;
    float *output = calloc(output_size, sizeof(float));

    ReLU(input, output, output_size);

    FILE *fp = fopen(output_path, "wb");
    fwrite(output, sizeof(float), output_size, fp);
    fclose(fp);

    free(input);
    return output;
}

static float *forward_Tanh(
    int layer,
    float *input,
    long batchSize,
    long nInputPlane,
    long inputWidth,
    long inputHeight)
{
    char output_path[256];
    sprintf(output_path, "bin/output_%i_test.bin", layer);

    long output_size = batchSize * nInputPlane * inputWidth * inputHeight;
    float *output = calloc(output_size, sizeof(float));

    Tanh(input, output, output_size);

    FILE *fp = fopen(output_path, "wb");
    fwrite(output, sizeof(float), output_size, fp);
    fclose(fp);

    free(input);
    return output;
}

int main(void)
{
    float *input_1 = calloc(64 * 100, sizeof(float));
    FILE *fp = fopen("bin/input_1.bin", "rb");
    fread(input_1, sizeof(float), 64 * 100, fp);
    fclose(fp);

    // (64, 100, 1, 1) -> (64, 512, 4, 4)
    float *output_1 = forward_SpatialFullConvolution(
        1, input_1, 64, 100, 1, 1, 512, 4, 4, 1, 1, 0, 0);

    // (64, 512, 4, 4) -> (64, 512, 4, 4)
    float *output_2 = forward_SpatialBatchNormalization(
        2, output_1, 64, 512, 4, 4);

    // (64, 512, 4, 4) -> (64, 512, 4, 4)
    float *output_3 = forward_ReLU(3, output_2, 64, 512, 4, 4);

    // (64, 512, 4, 4) -> (64, 256, 8, 8)
    float *output_4 = forward_SpatialFullConvolution(
        4, output_3, 64, 512, 4, 4, 256, 4, 4, 2, 2, 1, 1);

    // (64, 256, 8, 8) -> (64, 256, 8, 8)
    float *output_5 = forward_SpatialBatchNormalization(
        5, output_4, 64, 256, 8, 8);

    // (64, 256, 8, 8) -> (64, 256, 8, 8)
    float *output_6 = forward_ReLU(6, output_5, 64, 256, 8, 8);

    // (64, 256, 8, 8) -> (64, 128, 16, 16)
    float *output_7 = forward_SpatialFullConvolution(
        7, output_6, 64, 256, 8, 8, 128, 4, 4, 2, 2, 1, 1);

    // (64, 128, 16, 16) -> (64, 128, 16, 16)
    float *output_8 = forward_SpatialBatchNormalization(
        8, output_7, 64, 128, 16, 16);

    // (64, 128, 16, 16) -> (64, 128, 16, 16)
    float *output_9 = forward_ReLU(9, output_8, 64, 128, 16, 16);

    // (64, 128, 16, 16) -> (64, 64, 32, 32)
    float *output_10 = forward_SpatialFullConvolution(
        10, output_9, 64, 128, 16, 16, 64, 4, 4, 2, 2, 1, 1);

    // (64, 64, 32, 32) -> (64, 64, 32, 32)
    float *output_11 = forward_SpatialBatchNormalization(
        11, output_10, 64, 64, 32, 32);

    // (64, 64, 32, 32) -> (64, 64, 32, 32)
    float *output_12 = forward_ReLU(12, output_11, 64, 64, 32, 32);

    // (64, 64, 32, 32) -> (64, 3, 64, 64)
    float *output_13 = forward_SpatialFullConvolution(
        13, output_12, 64, 64, 32, 32, 3, 4, 4, 2, 2, 1, 1);

    // (64, 3, 64, 64) -> (64, 3, 64, 64)
    float *output_14 = forward_Tanh(14, output_13, 64, 3, 64, 64);
    free(output_14);

    return 0;
}
