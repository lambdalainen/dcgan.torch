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
    printf("elt %i\n", elt);
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
    printf("after gemm, output %p, output_n %p\n", output, output_n);

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

#if 0
static void layer_2(void)
{
}

static void layer_3(void)
{
}

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

    // float *weight_4;
    // float *weight_7;
    // float *weight_10;
    // float *weight_13;

    return 0;
}
