#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"
#include "col2im.h"

static int dilationW = 1;
static int dilationH = 1;

static void layer_1(
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
  long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1);
  long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1);

  // Define a buffer of ones, for bias accumulation
  int ones_n = outputHeight * outputWidth;
  float *ones = calloc(ones_n, sizeof(float));
  for (int i = 0; i < ones_n; i++)
      ones[i] = 1;

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
    long m = 8192;
    long n = 1;
    long k = 100;

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
}

static void layer_2(
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

static void layer_3(
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

    layer_1(input_1, output_1, weight_1, bias_1, columns_1,
            4, 4, 1, 1, 0, 0, 100, 512, 1, 1, 64);

    fp = fopen("bin/output_1_test.bin", "wb");
    fwrite(output_1, sizeof(float), 64 * 512 * 4 * 4, fp);
    fclose(fp);

    free(input_1);
    free(weight_1);
    free(bias_1);
    free(columns_1);



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

    layer_2(input_2, output_2, weight_2, bias_2,
            512, 4, 4, 64);

    fp = fopen("bin/output_2_test.bin", "wb");
    fwrite(output_2, sizeof(float), 64 * 512 * 4 * 4, fp);
    fclose(fp);

    free(input_2);
    free(weight_2);
    free(bias_2);



    // (64, 512, 4, 4)
    float *input_3 = output_2;
    // (64, 512, 4, 4)
    float *output_3 = calloc(64 * 512 * 4 * 4, sizeof(float));

    layer_3(input_3, output_3, 64 * 512 * 4 * 4);

    fp = fopen("bin/output_3_test.bin", "wb");
    fwrite(output_3, sizeof(float), 64 * 512 * 4 * 4, fp);
    fclose(fp);

    free(input_3);



    // (64, 512, 4, 4)
    float *input_4 = output_3;
    // float *weight_4;
    // float *weight_7;
    // float *weight_10;
    // float *weight_13;

    return 0;
}
