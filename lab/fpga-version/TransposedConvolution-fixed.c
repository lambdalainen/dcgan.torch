#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void sparse_gemm_fixed(long m, long n, long k,
                       uint8_t *a, long lda, uint8_t *b, long ldb,
                       long nOutputPlane,
                       long outputHeight, long outputWidth,
                       long inputHeight, long inputWidth,
                       int kH, int kW,
                       int padH, int padW,
                       int strideH, int strideW,
                       int dilationH, int dilationW,
                       int32_t* data_im);

static int dilationW = 1;
static int dilationH = 1;

static void SpatialFullConvolution_fixed(
    uint8_t *input,
    int32_t *output,
    uint8_t *weight,
    int32_t *bias,
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
    printf("m %li, k %li, n %li\n", m, k, n);

    // gemm & col2im combined
    sparse_gemm_fixed(
        m, n, k,
        input_n, m,
        weight, k,
        nOutputPlane,
        outputHeight, outputWidth,
        inputHeight, inputWidth,
        kH, kW, padH, padW, dH, dW,
        dilationH, dilationW,
        output_n
    );
  }
}

static int32_t *forward_SpatialFullConvolution(
    uint8_t *input,
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
    long outputHeight = (inputHeight - 1) * dH - 2*padH + (dilationH * (kH - 1) + 1);
    long outputWidth  = (inputWidth - 1) * dW - 2*padW + (dilationW * (kW - 1) + 1);
    long output_size = batchSize * nOutputPlane * outputWidth * outputHeight;
    int32_t *output = calloc(output_size, sizeof(int32_t));
    printf("output_size: %li\n", output_size);

    // weight: nInputPlane * nOutputPlane * kW * kH;
    uint8_t weight[8] = {1, 2, 3, 4, 1, 2, 3, 4};
    int32_t bias[2] = {0.0f, 0.0f};

    SpatialFullConvolution_fixed(
        input, output, weight, bias,
        batchSize, nInputPlane, inputWidth, inputHeight, nOutputPlane,
        kW, kH, dW, dH, padW, padH);

    return output;
}

int main(void)
{
    // 2 x 3 x 3
    uint8_t input_1[18] =
        { 1, 2, 3,
          3, 2, 1,
          1, 2, 3,
          1, 2, 3,
          3, 2, 1,
          1, 2, 3 };

    int32_t *output_1 = forward_SpatialFullConvolution(
        input_1, 1, 2, 3, 3, 1, 2, 2, 2, 2, 1, 1);
    
    printf("Result: \n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%i ", output_1[i * 4 + j]);
        }
        printf("\n");
    }

    free(output_1);
    return 0;
}
