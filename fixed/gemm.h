#ifndef __GEMM_H__
#define __GEMM_H__

struct Q {
    float *f;     // original data
    float *f2;    // for fusion
    uint8_t *q;   // quantized data
    int32_t *q32; // quantized data
    float min;
    float max;
    float s;   // scale
    uint8_t z; // zero point
};

void gemm(char transa, char transb,
          long m, long n, long k,
          float alpha, float *a, long lda, float *b, long ldb,
          float beta, float *c, long ldc);

void gemm_fixed(
          char transa, char transb,
          long m, long n, long k,
          uint8_t alpha, uint8_t *a, long lda, uint8_t *b, long ldb,
          uint8_t beta, int32_t *c, long ldc,
          struct Q *lhs, struct Q *rhs, struct Q *res);

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
                       struct Q *lhs, struct Q *rhs, struct Q *res, int layer);

#endif
