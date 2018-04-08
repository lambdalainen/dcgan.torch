#ifndef __GEMM_H__
#define __GEMM_H__

void gemm(char transa, char transb,
          long m, long n, long k,
          float alpha, float *a, long lda, float *b, long ldb,
          float beta, float *c, long ldc);

void sparse_gemm(long m, long n, long k,
                 float *a, long lda, float *b, long ldb,
                 long nOutputPlane,
                 long outputHeight, long outputWidth,
                 long inputHeight, long inputWidth,
                 int kH, int kW,
                 int padH, int padW,
                 int strideH, int strideW,
                 int dilationH, int dilationW,
                 float* data_im);

#endif
