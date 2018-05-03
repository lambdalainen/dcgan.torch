#include <stdio.h>
#include <stdint.h>
#include "gemm.h"

#define save_txt(path, data, count) do { \
    FILE *fp = fopen(path, "wb"); \
    for (long i = 0; i < count; i++) \
        fprintf(fp, "%x\n", data[i]); \
    fclose(fp); \
} while (0)

// Why this works:
// 0. recognize that the i & j for-loops in gemm can be swapped
// 1. recognize that the j for-loop in gemm can be splitted into 3 nested for-loops
// 2. recognize that the nested for-loops for h_col & w_col is equivalent to the i for-loop in gemm
// 3. data_col[(j * inputHeight + h_col) * inputWidth + w_col]; -> expaned this, you get
//    (j * inputHeight * inputWidth) + (h_col * inputWidth + w_col)
//    the first part is equivalent to j * ldc, the second equivalent to i
//    so this is the same as c[j*ldc+i] in gemm
void sparse_gemm_fixed(long m, long n, long k,
                       uint8_t *a, long lda, uint8_t *b, long ldb,
                       long nOutputPlane,
                       long outputHeight, long outputWidth,
                       long inputHeight, long inputWidth,
                       int kH, int kW,
                       int padH, int padW,
                       int strideH, int strideW,
                       int dilationH, int dilationW,
                       int32_t* data_im,
                       struct Q *lhs, struct Q *rhs, struct Q *res, int layer)
{
    // https://github.com/google/gemmlowp/blob/master/doc/low-precision.md#efficient-handling-of-offsets
    // Term 2: lhs_offset * P * rhs: sum up each column of rhs to make a row vector,
    // scale by lhs zero offset, and add this vector to each row of the result
    int32_t row_vector[n];

    for (long j = 0; j < n; j++)
    {
        int32_t sum = 0;
        for (long l = 0; l < k; l++)
            sum += b[l*ldb+j]; // b is in row-major order
        row_vector[j] = - lhs->z * sum;
    }

    if (layer) {
        char path[256];
        sprintf(path, "../bin/row_vec_%i_int32.mem", layer);
        save_txt(path, row_vector, n);
    }

    // Term 3: lhs * rhs_offset * Q: sum each row of lhs to make a column-vector,
    // scale by rhs zero offset, and add this vector to each column of the result
    int32_t column_vector[m];

    for (long i = 0; i < m; i++)
    {
        int32_t sum = 0;
        for (long l = 0; l < k; l++)
            sum += a[l*lda+i]; // a is in column major order
        column_vector[i] = - rhs->z * sum;
    }

    // Term 4: lhs_offset * rhs_offset * P * Q (constant): same as lhs_offset * rhs_offset * depth,
    // where depth is the number of columns of the lhs, this constant is added to
    // each element of the result
    int32_t term_4 = lhs->z * rhs->z * k;

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
                            int32_t sum = 0;
                            for (long l = 0; l < k; l++)
                                sum += a[l*lda + i] * b[l*ldb+j];
                            sum += row_vector[j]; // Term 2
                            sum += column_vector[i]; // Term 3
                            sum += term_4; // Term 4
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

