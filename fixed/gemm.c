#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "gemm.h"

void gemm(char transa, char transb,
          long m, long n, long k,
          float alpha, float *a, long lda, float *b, long ldb,
          float beta, float *c, long ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    ldc = m;

  if(transa_)
  {
    if(m == 1)
      lda = k;
  }
  else
  {
    if(k == 1)
      lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      ldb = n;
  }
  else
  {
    if(n == 1)
      ldb = k;
  }

  {
    long i, j, l;
#if 0
    if(!transa_ && !transb_)
    {
      float *a_ = a;
      for(i = 0; i < m; i++)
      {
        float *b_ = b;
        for(j = 0; j < n; j++)
        {
          float sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else if(transa_ && !transb_)
    {
      float *a_ = a;
      for(i = 0; i < m; i++)
      {
        float *b_ = b;
        for(j = 0; j < n; j++)
        {
          float sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
    else if(!transa_ && transb_)
    {
#endif
      float *a_ = a;
      for(i = 0; i < m; i++)
      {
        float *b_ = b;
        for(j = 0; j < n; j++)
        {
          float sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
#if 0
    }
    else
    {
      float *a_ = a;
      for(i = 0; i < m; i++)
      {
        float *b_ = b;
        for(j = 0; j < n; j++)
        {
          float sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
#endif
  }
}

void gemm_fixed(char transa, char transb,
          long m, long n, long k,
          uint8_t alpha, uint8_t *a, long lda, uint8_t *b, long ldb,
          uint8_t beta, int32_t *c, long ldc,
          struct Q *lhs, struct Q *rhs, struct Q *res)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    ldc = m;

  if(transa_)
  {
    if(m == 1)
      lda = k;
  }
  else
  {
    if(k == 1)
      lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      ldb = n;
  }
  else
  {
    if(n == 1)
      ldb = k;
  }

  {
    long i, j, l;
#if 0
    if(!transa_ && !transb_)
    {
      float *a_ = a;
      for(i = 0; i < m; i++)
      {
        float *b_ = b;
        for(j = 0; j < n; j++)
        {
          float sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else if(transa_ && !transb_)
    {
      float *a_ = a;
      for(i = 0; i < m; i++)
      {
        float *b_ = b;
        for(j = 0; j < n; j++)
        {
          float sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l];
          b_ += ldb;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
    else if(!transa_ && transb_)
    {
#endif
      uint8_t *a_ = a;
      for(i = 0; i < m; i++)
      {
        uint8_t *b_ = b;
        for(j = 0; j < n; j++)
        {
          int32_t sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }

#if 1
      // https://github.com/google/gemmlowp/blob/master/doc/low-precision.md#efficient-handling-of-offsets
      // Term 2: lhs_offset * P * rhs: sum up each column of rhs to make a row vector,
      // scale by lhs zero offset, and add this vector to each row of the result
      int32_t row_vector[n];

      for (j = 0; j < n; j++)
      {
          int32_t sum = 0;
          for (l = 0; l < k; l++)
              sum += b[l*ldb+j]; // b is in row-major order
          row_vector[j] = - lhs->z * sum;
      }

      for (i = 0; i < m; i++)
      {
          for (j = 0; j < n; j++)
              c[j*ldc+i] += row_vector[j];
      }

      // Term 3: lhs * rhs_offset * Q: sum each row of lhs to make a column-vector,
      // scale by rhs zero offset, and add this vector to each column of the result
      int32_t column_vector[m];

      for (i = 0; i < m; i++)
      {
          int32_t sum = 0;
          for (l = 0; l < k; l++)
              sum += a[l*lda+i]; // a is in column major order
          column_vector[i] = - rhs->z * sum;
      }

      for (j = 0; j < n; j++)
      {
          for (i = 0; i < m; i++)
              c[j*ldc+i] += column_vector[i];
      }

      // Term 4: lhs_offset * rhs_offset * P * Q (constant): same as lhs_offset * rhs_offset * depth,
      // where depth is the number of columns of the lhs, this constant is added to
      // each element of the result
      int32_t term_4 = lhs->z * rhs->z * k;

      for (i = 0; i < m; i++)
      {
          for (j = 0; j < n; j++)
              c[j*ldc+i] += term_4;
      }

#if 0 // we choose to scale down after col2im and bias addition
      // scale down to uint8_t
      float scale = lhs->s * rhs->s / res->s;

      for (i = 0; i < m; i++)
      {
          for (j = 0; j < n; j++) {
              long idx = j*ldc+i;
              int32_t result = round(c[idx] * scale) + res->z;
              if (result < 0)
                  result = 0;
              else if (result > 255)
                  result = 255;
              c[idx] = result;
          }
      }
#endif
#endif

#if 0
    }
    else
    {
      float *a_ = a;
      for(i = 0; i < m; i++)
      {
        float *b_ = b;
        for(j = 0; j < n; j++)
        {
          float sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l*ldb];
          b_++;
	  if (beta == 0)
	    c[j*ldc+i] = alpha*sum;
	  else
	    c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
#endif
  }
}

