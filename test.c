#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"
#include "col2im.h"

static int dilationW = 1;
static int dilationH = 1;
static int spatial_full_conv_layer;

static void SpatialFullConvolution(
    float *input,
    float *output,
    float *weight,
    float *bias,
    float *columns,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    int nInputPlane,
    int nOutputPlane,
    long inputHeight,
    long inputWidth,
    long batchSize)
{
  spatial_full_conv_layer++;

  long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1);
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1);

  // Define a buffer of ones, for bias accumulation
  int ones_n = outputHeight * outputWidth;
  float *ones = calloc(ones_n, sizeof(float));
  for (int i = 0; i < ones_n; i++)
      ones[i] = 1;

  printf("kW %i, kH %i, dW %i, dH %i, padW %i, padH %i, dilationW %i, dilationH %i\n",
          kW, kH, dW, dH, padW, padH, dilationW, dilationH);
  printf("nInputPlane %i, nOutputPlane %i\n", nInputPlane, nOutputPlane);
  printf("inputHeight %li, inputWidth %li, outputHeight %li, outputWidth %li, batchSize %li\n",
          inputHeight, inputWidth, outputHeight, outputWidth, batchSize);

  // Helpers
  float *input_n;
  float *output_n;

  for (int elt = 0; elt < batchSize; elt++) {
    //printf("elt %i\n", elt);
    // Matrix mulitply per output:
    // Note: input + 1 means addr + sizeof(float)
    input_n = input + elt * nInputPlane * inputHeight * inputWidth;
    output_n = output + elt * nOutputPlane * outputHeight * outputWidth;

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    // long m = weight->size[1] * weight->size[2] * weight->size[3];
    // long n = columns->size[1];
    // long k = weight->size[0];
    long m = nOutputPlane * kW * kH;
    long n = inputHeight * inputWidth;
    long k = nInputPlane;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    gemm(
        'n', 't',
        n, m, k,
        1,
        input_n, n,
        weight, m,
        0,
        columns, n
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
    long m_ = nOutputPlane;
    long n_ = outputHeight * outputWidth;
    long k_ = 1;

    gemm(
        't', 'n',
        n_, m_, k_,
        1,
        ones, k_,
        bias, k_,
        1,
        output_n, n_
    );
  }

  free(ones);
  printf("### finished: spatial_full_conv_layer %i\n\n", spatial_full_conv_layer);
}

static void SpatialBatchNormalization(
  float *input,
  float *output,
  float *weight,
  float *bias,
  //float *running_mean,
  //float *running_var
  //float *save_mean,
  //float *save_std,
  int nInputPlane, // input->size[1]
  long inputWidth,
  long inputHeight,
  long batchSize)
{
  //double momentum = 0.1;
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
    //THTensor_(set1d)(save_mean, f, (float) mean);

    // compute variance per input
    sum = 0;
    for (int i = 0; i < batchSize; i++) {
        float *plane_ptr = in + i * nInputPlane * input_plane_stride;
        for (float *p = plane_ptr; p < (plane_ptr + input_plane_stride); p++)
            sum += (*p - mean) * (*p - mean);
    }

    invstd = (float) (1 / sqrt(sum/n + eps));
    //THTensor_(set1d)(save_std, f, (float) invstd);

    // update running averages
    //THTensor_(set1d)(running_mean, f,
    //  (float) (momentum * mean + (1 - momentum) * THTensor_(get1d)(running_mean, f)));

    //double unbiased_var = sum / (n - 1);
    //THTensor_(set1d)(running_var, f,
    //  (float) (momentum * unbiased_var + (1 - momentum) * THTensor_(get1d)(running_var, f)));

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
}

static void ReLU(
    float *input,
    float *output,
    long size)
{
    float *p, *q;
    for (p = input, q = output;
         p < (input + size) && q < (output + size);
         p++, q++) {
        *q = *p > 0 ? *p : 0;
    }
}

static void Tanh(
  float *input,
  float *output,
  long size)
{
    float *p, *q;
    for (p = input, q = output;
         p < (input + size) && q < (output + size);
         p++, q++) {
        *q = tanh(*p);
    }
    printf("tanh: first 5 outputs: ");
    for (int i = 0; i < 5; i++)
        printf("%f ", output[i]);
    printf("\n");
}

#if 0
static void layer_4(void)
{
}

static void layer_5(void)
{
}

static void layer_6(void)
{
}

static void layer_7(void)
{
}

static void layer_8(void)
{
}

static void layer_9(void)
{
}

static void layer_10(void)
{
}

static void layer_11(void)
{
}

static void layer_12(void)
{
}

static void layer_13(void)
{
}

static void layer_14(void)
{
}
#endif

int main(void)
{
    /* ----- (1): nn.SpatialFullConvolution(100 -> 512, 4x4) ----- */

    // (64, 100, 1, 1)
    float *input_1 = calloc(64 * 100, sizeof(float));
    // (64, 512, 4, 4)
    float *output_1 = calloc(64 * 512 * 4 * 4, sizeof(float));
    // (100, 512, 4, 4)
    float *weight_1 = calloc(100 * 512 * 4 * 4, sizeof(float));
    // (512)
    float *bias_1 = calloc(512, sizeof(float));
    // (8192, 1)
    // THTensor_(resize2d)(columns, nOutputPlane*kW*kH, inputHeight*inputWidth);
    float *columns_1 = calloc(8192, sizeof(float));

    FILE *fp = fopen("bin/input_1.bin", "rb");
    fread(input_1, sizeof(float), 64 * 100, fp);
    fclose(fp);
    fp = fopen("bin/weight_1.bin", "rb");
    fread(weight_1, sizeof(float), 100 * 512 * 4 * 4, fp);
    fclose(fp);
    fp = fopen("bin/bias_1.bin", "rb");
    fread(bias_1, sizeof(float), 512, fp);
    fclose(fp);

    SpatialFullConvolution(input_1, output_1, weight_1, bias_1, columns_1,
            4, 4, 1, 1, 0, 0, 100, 512, 1, 1, 64);

    fp = fopen("bin/output_1_test.bin", "wb");
    fwrite(output_1, sizeof(float), 64 * 512 * 4 * 4, fp);
    fclose(fp);

    free(input_1);
    free(weight_1);
    free(bias_1);
    free(columns_1);


    /* ----- (2): nn.SpatialBatchNormalization (4D) (512) ----- */

    // (64, 512, 4, 4)
    float *input_2 = output_1;
    // (64, 512, 4, 4)
    float *output_2 = calloc(64 * 512 * 4 * 4, sizeof(float));
    // (512)
    float *weight_2 = calloc(512, sizeof(float));
    // (512)
    float *bias_2 = calloc(512, sizeof(float));

    fp = fopen("bin/weight_2.bin", "rb");
    fread(weight_2, sizeof(float), 512, fp);
    fclose(fp);
    fp = fopen("bin/bias_2.bin", "rb");
    fread(bias_2, sizeof(float), 512, fp);
    fclose(fp);

    SpatialBatchNormalization(input_2, output_2, weight_2, bias_2,
            512, 4, 4, 64);

    fp = fopen("bin/output_2_test.bin", "wb");
    fwrite(output_2, sizeof(float), 64 * 512 * 4 * 4, fp);
    fclose(fp);

    free(input_2);
    free(weight_2);
    free(bias_2);


    /* ----- (3): nn.ReLU ----- */

    // (64, 512, 4, 4)
    float *input_3 = output_2;
    // (64, 512, 4, 4)
    float *output_3 = calloc(64 * 512 * 4 * 4, sizeof(float));

    ReLU(input_3, output_3, 64 * 512 * 4 * 4);

    fp = fopen("bin/output_3_test.bin", "wb");
    fwrite(output_3, sizeof(float), 64 * 512 * 4 * 4, fp);
    fclose(fp);

    free(input_3);


    /* ----- (4): nn.SpatialFullConvolution(512 -> 256, 4x4, 2,2, 1,1) ----- */

    // (64, 512, 4, 4)
    float *input_4 = output_3;
    // (64, 256, 8, 8)
    float *output_4 = calloc(64 * 256 * 8 * 8, sizeof(float));
    // (512, 256, 4, 4)
    float *weight_4 = calloc(512 * 256 * 4 * 4, sizeof(float));
    // (256)
    float *bias_4 = calloc(256, sizeof(float));
    // (4096, 16)
    float *columns_4 = calloc(4096 * 16, sizeof(float));

    fp = fopen("bin/weight_4.bin", "rb");
    fread(weight_4, sizeof(float), 512 * 256 * 4 * 4, fp);
    fclose(fp);
    fp = fopen("bin/bias_4.bin", "rb");
    fread(bias_4, sizeof(float), 256, fp);
    fclose(fp);

    SpatialFullConvolution(input_4, output_4, weight_4, bias_4, columns_4,
            4, 4, 2, 2, 1, 1, 512, 256, 4, 4, 64);

    fp = fopen("bin/output_4_test.bin", "wb");
    fwrite(output_4, sizeof(float), 64 * 256 * 8 * 8, fp);
    fclose(fp);

    free(input_4);
    free(weight_4);
    free(bias_4);
    free(columns_4);


    /* ----- (5): nn.SpatialBatchNormalization (4D) (256) ----- */

    // (64, 256, 8, 8)
    float *input_5 = output_4;
    // (64, 256, 8, 8)
    float *output_5 = calloc(64 * 256 * 8 * 8, sizeof(float));
    // (256)
    float *weight_5 = calloc(256, sizeof(float));
    // (256)
    float *bias_5 = calloc(256, sizeof(float));

    fp = fopen("bin/weight_5.bin", "rb");
    fread(weight_5, sizeof(float), 256, fp);
    fclose(fp);
    fp = fopen("bin/bias_5.bin", "rb");
    fread(bias_5, sizeof(float), 256, fp);
    fclose(fp);

    SpatialBatchNormalization(input_5, output_5, weight_5, bias_5,
            256, 8, 8, 64);

    fp = fopen("bin/output_5_test.bin", "wb");
    fwrite(output_5, sizeof(float), 64 * 256 * 8 * 8, fp);
    fclose(fp);

    free(input_5);
    free(weight_5);
    free(bias_5);


    /* ----- (6): nn.ReLU ----- */

    // (64, 256, 8, 8)
    float *input_6 = output_5;
    // (64, 256, 8, 8)
    float *output_6 = calloc(64 * 256 * 8 * 8, sizeof(float));

    ReLU(input_6, output_6, 64 * 256 * 8 * 8);

    fp = fopen("bin/output_6_test.bin", "wb");
    fwrite(output_6, sizeof(float), 64 * 256 * 8 * 8, fp);
    fclose(fp);

    free(input_6);


    /* ----- (7): nn.SpatialFullConvolution(256 -> 128, 4x4, 2,2, 1,1) ----- */

    // (64, 256, 8, 8)
    float *input_7 = output_6;
    // (64, 128, 16, 16)
    float *output_7 = calloc(64 * 128 * 16 * 16, sizeof(float));
    // (256, 128, 4, 4)
    float *weight_7 = calloc(256 * 128 * 4 * 4, sizeof(float));
    // (128)
    float *bias_7 = calloc(128, sizeof(float));
    // (2048, 64)
    float *columns_7 = calloc(2048 * 64, sizeof(float));

    fp = fopen("bin/weight_7.bin", "rb");
    fread(weight_7, sizeof(float), 256 * 128 * 4 * 4, fp);
    fclose(fp);
    fp = fopen("bin/bias_7.bin", "rb");
    fread(bias_7, sizeof(float), 128, fp);
    fclose(fp);

    SpatialFullConvolution(input_7, output_7, weight_7, bias_7, columns_7,
            4, 4, 2, 2, 1, 1, 256, 128, 8, 8, 64);

    fp = fopen("bin/output_7_test.bin", "wb");
    fwrite(output_7, sizeof(float), 64 * 128 * 16 * 16, fp);
    fclose(fp);

    free(input_7);
    free(weight_7);
    free(bias_7);
    free(columns_7);


    /* ----- (8): nn.SpatialBatchNormalization (4D) (128) ----- */

    // (64, 128, 16, 16)
    float *input_8 = output_7;
    // (64, 128, 16, 16)
    float *output_8 = calloc(64 * 128 * 16 * 16, sizeof(float));
    // (128)
    float *weight_8 = calloc(128, sizeof(float));
    // (128)
    float *bias_8 = calloc(128, sizeof(float));

    fp = fopen("bin/weight_8.bin", "rb");
    fread(weight_8, sizeof(float), 128, fp);
    fclose(fp);
    fp = fopen("bin/bias_8.bin", "rb");
    fread(bias_8, sizeof(float), 128, fp);
    fclose(fp);

    SpatialBatchNormalization(input_8, output_8, weight_8, bias_8,
            128, 16, 16, 64);

    fp = fopen("bin/output_8_test.bin", "wb");
    fwrite(output_8, sizeof(float), 64 * 128 * 16 * 16, fp);
    fclose(fp);

    free(input_8);
    free(weight_8);
    free(bias_8);


    /* ----- (9): nn.ReLU ----- */

    // (64, 128, 16, 16)
    float *input_9 = output_8;
    // (64, 128, 16, 16)
    float *output_9 = calloc(64 * 128 * 16 * 16, sizeof(float));

    ReLU(input_9, output_9, 64 * 128 * 16 * 16);

    fp = fopen("bin/output_9_test.bin", "wb");
    fwrite(output_9, sizeof(float), 64 * 128 * 16 * 16, fp);
    fclose(fp);

    free(input_9);


    /* ----- (10): nn.SpatialFullConvolution(128 -> 64, 4x4, 2,2, 1,1) ----- */

    // (64, 128, 16, 16)
    float *input_10 = output_9;
    // (64, 64, 32, 32)
    float *output_10 = calloc(64 * 64 * 32 * 32, sizeof(float));
    // (128, 64, 4, 4)
    float *weight_10 = calloc(128 * 64 * 4 * 4, sizeof(float));
    // (64)
    float *bias_10 = calloc(64, sizeof(float));
    // (1024, 256)
    float *columns_10 = calloc(1024 * 256, sizeof(float));

    fp = fopen("bin/weight_10.bin", "rb");
    fread(weight_10, sizeof(float), 128 * 64 * 4 * 4, fp);
    fclose(fp);
    fp = fopen("bin/bias_10.bin", "rb");
    fread(bias_10, sizeof(float), 64, fp);
    fclose(fp);

    SpatialFullConvolution(input_10, output_10, weight_10, bias_10, columns_10,
            4, 4, 2, 2, 1, 1, 128, 64, 16, 16, 64);

    fp = fopen("bin/output_10_test.bin", "wb");
    fwrite(output_10, sizeof(float), 64 * 64 * 32 * 32, fp);
    fclose(fp);

    free(input_10);
    free(weight_10);
    free(bias_10);
    free(columns_10);


    /* ----- (11): nn.SpatialBatchNormalization (4D) (64) ----- */

    // (64, 64, 32, 32)
    float *input_11 = output_10;
    // (64, 64, 32, 32)
    float *output_11 = calloc(64 * 64 * 32 * 32, sizeof(float));
    // (64)
    float *weight_11 = calloc(64, sizeof(float));
    // (64)
    float *bias_11 = calloc(64, sizeof(float));

    fp = fopen("bin/weight_11.bin", "rb");
    fread(weight_11, sizeof(float), 64, fp);
    fclose(fp);
    fp = fopen("bin/bias_11.bin", "rb");
    fread(bias_11, sizeof(float), 64, fp);
    fclose(fp);

    SpatialBatchNormalization(input_11, output_11, weight_11, bias_11,
            64, 32, 32, 64);

    fp = fopen("bin/output_11_test.bin", "wb");
    fwrite(output_11, sizeof(float), 64 * 64 * 32 * 32, fp);
    fclose(fp);

    free(input_11);
    free(weight_11);
    free(bias_11);


    /* ----- (12): nn.ReLU ----- */

    // (64, 64, 32, 32)
    float *input_12 = output_11;
    // (64, 64, 32, 32)
    float *output_12 = calloc(64 * 64 * 32 * 32, sizeof(float));

    ReLU(input_12, output_12, 64 * 64 * 32 * 32);

    fp = fopen("bin/output_12_test.bin", "wb");
    fwrite(output_12, sizeof(float), 64 * 64 * 32 * 32, fp);
    fclose(fp);

    free(input_12);
    

    /* ----- (13): nn.SpatialFullConvolution(64 -> 3, 4x4, 2,2, 1,1) ----- */

    // (64, 64, 32, 32)
    float *input_13 = output_12;
    // (64, 3, 64, 64)
    float *output_13 = calloc(64 * 3 * 64 * 64, sizeof(float));
    // (64, 3, 4, 4)
    float *weight_13 = calloc(64 * 3 * 4 * 4, sizeof(float));
    // (3)
    float *bias_13 = calloc(3, sizeof(float));
    // (48, 1024)
    float *columns_13 = calloc(48 * 1024, sizeof(float));

    fp = fopen("bin/weight_13.bin", "rb");
    fread(weight_13, sizeof(float), 64 * 3 * 4 * 4, fp);
    fclose(fp);
    fp = fopen("bin/bias_13.bin", "rb");
    fread(bias_13, sizeof(float), 3, fp);
    fclose(fp);

    SpatialFullConvolution(input_13, output_13, weight_13, bias_13, columns_13,
            4, 4, 2, 2, 1, 1, 64, 3, 32, 32, 64);

    fp = fopen("bin/output_13_test.bin", "wb");
    fwrite(output_13, sizeof(float), 64 * 3 * 64 * 64, fp);
    fclose(fp);

    free(input_13);
    free(weight_13);
    free(bias_13);
    free(columns_13);


    /* ----- (14): nn.Tanh ----- */

    // (64, 3, 64, 64)
    float *input_14 = output_13;
    // (64, 3, 64, 64)
    float *output_14 = calloc(64 * 3 * 64 * 64, sizeof(float));

    Tanh(input_14, output_14, 64 * 3 * 64 * 64);

    fp = fopen("bin/output_14_test.bin", "wb");
    fwrite(output_14, sizeof(float), 64 * 3 * 64 * 64, fp);
    fclose(fp);

    free(input_14);

    return 0;
}
