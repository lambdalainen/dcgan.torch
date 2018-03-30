#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void gemm(int debug, char transa, char transb,
          long m, long n, long k,
          float alpha, float *a, long lda, float *b, long ldb,
          float beta, float *c, long ldc);
void col2im(int debug, const float* data_col, const int channels,
            const int height, const int width,
            const int output_height, const int output_width,
            const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w,
            const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w,
            float* data_im);

static int dilationW = 1;
static int dilationH = 1;

static void SpatialFullConvolution(
    int debug,
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
        debug,
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
      debug,
      columns,
      nOutputPlane, outputHeight, outputWidth, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      dilationH, dilationW,
      output_n
    );

    // Do Bias after:
#if 1
    long m_ = nOutputPlane;
    long n_ = outputHeight * outputWidth;
    long k_ = 1;

    gemm(
        0,
        't', 'n',
        n_, m_, k_,
        1,
        ones, k_,
        bias, k_,
        1,
        output_n, n_
    );
#endif
  }

  free(ones);
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
    printf("sum %f, mean %f\n", sum, mean);

    // compute variance per input
    sum = 0;
    for (int i = 0; i < batchSize; i++) {
        float *plane_ptr = in + i * nInputPlane * input_plane_stride;
        for (float *p = plane_ptr; p < (plane_ptr + input_plane_stride); p++)
            sum += (*p - mean) * (*p - mean);
    }

    invstd = (float) (1 / sqrt(sum/n + eps));
    printf("sum %f, invstd %f\n", sum, invstd);

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

static float *forward_SpatialFullConvolution(
    int debug,
    float *input,
    float *weight,
    float *bias,
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
    float *output = calloc(output_size, sizeof(float));

    // columns: (nOutputPlane*kW*kH, inputHeight*inputWidth)
    float *columns = calloc(nOutputPlane * kW * kH * inputHeight * inputWidth, sizeof(float));

    SpatialFullConvolution(
        debug, input, output, weight, bias, columns,
        batchSize, nInputPlane, inputWidth, inputHeight, nOutputPlane,
        kW, kH, dW, dH, padW, padH);

    free(columns);

    return output;
}

static float *forward_SpatialBatchNormalization(
    float *input,
    long batchSize,
    long nInputPlane,
    long inputWidth,
    long inputHeight)
{
    // same shape for input and output
    long output_size = batchSize * nInputPlane * inputWidth * inputHeight;
    float *output = calloc(output_size, sizeof(float));
    float weight[1] = {1.0f};
    float bias[1] = {0.0f};

    SpatialBatchNormalization(
        input, output, weight, bias,
        batchSize, nInputPlane, inputWidth, inputHeight);

    free(input);
    return output;
}

int main(void)
{
    float input_1[9] = {1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f};
    // weight: nInputPlane * nOutputPlane * kW * kH;
    float weight_1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float bias_1[1] = {0.0f};

    printf("SpatialFullConvolution:\n");
    float *output_1 = forward_SpatialFullConvolution(
        1, input_1, weight_1, bias_1,
        1, 1, 3, 3, 1, 2, 2, 2, 2, 1, 1);
    
    printf("Result after SpatialFullConvolution: \n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.2f ", output_1[i * 4 + j]);
        }
        printf("\n");
    }

    printf("\nSpatialBatchNormalization:\n");
    float *output_2 = forward_SpatialBatchNormalization(
        output_1, 1, 1, 4, 4);

    printf("Result after SpatialBatchNormalization: \n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", output_2[i * 4 + j]);
        }
        printf("\n");
    }

    printf("\nNow, solve linear equations\n");
    float w = (output_2[1] - output_2[0]) / (output_1[1] - output_1[0]);
    float b = output_2[0] - output_1[0] * w;
    printf("w %f, b %f\n", w, b);

    float weight_bn[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float bias_bn[1] = {0.0f};

    printf("Updated weights: ");
    for (int i = 0; i < 4; i++) {
        weight_bn[i] *= w;
        printf("%f ", weight_bn[i]);
    }
    bias_bn[0] = b;
    printf("\nUpdated bias: %f\n", bias_bn[0]);

    printf("\nSpatialFullConvolution:\n");
    float *output_bn = forward_SpatialFullConvolution(
        0, input_1, weight_bn, bias_bn,
        1, 1, 3, 3, 1, 2, 2, 2, 2, 1, 1);
    
    printf("Result after SpatialFullConvolution with SpatialBatchNormalization blended in: \n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", output_bn[i * 4 + j]);
        }
        printf("\n");
    }
    printf("As can be seen, the results are very close but not exactly the same\n");

    free(output_2);
    free(output_bn);
    return 0;
}
