#include <stdio.h>
#include "gemm.h"

// Why this works:
// 0. recognize that the i & j for-loops in gemm can be swapped
// 1. recognize that the nested for-loops for h_col & w_col is equivalent to the i for-loop in gemm
// 2. data_col[(j * inputHeight + h_col) * inputWidth + w_col]; -> expaned this, you get
//    (j * inputHeight * inputWidth) + (h_col * inputWidth + w_col)
//    the first part is equivalent to j * ldc, the second equivalent to i
//    so this is the same as c[j*ldc+i] in gemm
// Notice that m is not actually used, it's just here for clarity.
void sparse_gemm(long m, long n, long k,
                 float *a, long lda, float *b, long ldb,
                 long outputHeight, long outputWidth,
                 long inputHeight, long inputWidth,
                 int kH, int kW,
                 int padH, int padW,
                 int strideH, int strideW,
                 int dilationH, int dilationW,
                 float* data_im)
{
  // n = nOutputPlane * kH * kW;
  for (long j = 0; j < n; j++)
  {
    int w_offset = j % kW;
    int h_offset = (j / kW) % kH;
    int c_im = j / kH / kW;

    // the i-loop splitted into nested loops
    long i = 0;
    for (long h_col = 0; h_col < inputHeight; ++h_col) {
      for (long w_col = 0; w_col < inputWidth; ++w_col) {
        int h_im = h_col * strideH - padH + h_offset * dilationH;
        int w_im = w_col * strideW - padW + w_offset * dilationW;
        if (h_im >= 0 && h_im < outputHeight && w_im >= 0 && w_im < outputWidth) {
          float sum = 0;
          for (long l = 0; l < k; l++)
            sum += a[l*lda + i] * b[l*ldb+j];
          data_im[(c_im * outputHeight + h_im) * outputWidth + w_im] += sum;
        }
        i++;
      }
    }
    // i would equal to m here
  }
}

