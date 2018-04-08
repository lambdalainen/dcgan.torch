#include <stdio.h>
#include "gemm.h"

// Why this works:
// 0. recognize that the i & j for-loops in gemm can be swapped
// 1. recognize that the j for-loop in gemm can be splitted into 3 nested for-loops
// 2. recognize that the nested for-loops for h_col & w_col is equivalent to the i for-loop in gemm
// 3. data_col[(j * inputHeight + h_col) * inputWidth + w_col]; -> expaned this, you get
//    (j * inputHeight * inputWidth) + (h_col * inputWidth + w_col)
//    the first part is equivalent to j * ldc, the second equivalent to i
//    so this is the same as c[j*ldc+i] in gemm
// Notice that m and n are not actually used, they are just here for clarity.
void sparse_gemm(long m, long n, long k,
                 float *a, long lda, float *b, long ldb,
                 long nOutputPlane,
                 long outputHeight, long outputWidth,
                 long inputHeight, long inputWidth,
                 int kH, int kW,
                 int padH, int padW,
                 int strideH, int strideW,
                 int dilationH, int dilationW,
                 float* data_im)
{
    // the j-loop splitted into 3 nested loops (j from 0 to n = nOutputPlane * kH * kW)
    long j = 0;
    for (int c_im = 0; c_im < nOutputPlane; c_im++) {
        for (int h_offset = 0; h_offset < kH; h_offset++) {
            for (int w_offset = 0; w_offset < kW; w_offset++) {

                // the i-loop splitted into 2 nested loops (i from 0 to m = inputHeight * inputWidth)
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
                j++;
            }
        }
    }
  // j would equal to n here
}

