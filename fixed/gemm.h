#ifndef __GEMM_H__
#define __GEMM_H__

void gemm(char transa, char transb,
          long m, long n, long k,
          float alpha, float *a, long lda, float *b, long ldb,
          float beta, float *c, long ldc);

#endif
