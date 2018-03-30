#include <stdio.h>

void gemm(int debug, char transa, char transb,
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
      if (debug)
        printf("--- gemm: m %li, n %li, k %li, lda %li, ldb %li, ldc %li\n", m, n, k, lda, ldb, ldc);
      float *a_ = a;
      for(i = 0; i < m; i++)
      {
        float *b_ = b;
        for(j = 0; j < n; j++)
        {
          float sum = 0;
          for(l = 0; l < k; l++)
          {
            if (debug) {
              printf("--- gemm: i %li, j %li, l %li, l*lda %li, l*ldb %li, a_[l*lda] %.2f, b_[l*ldb] %.2f\n",
                     i, j, l, l*lda, l*ldb, a_[l*lda], b_[l*ldb]);
            }
            sum += a_[l*lda]*b_[l*ldb];
          }
          b_++;
	  if (beta == 0)
          {
            if (debug)
              printf("--- gemm: j*ldc+i %li, sum %.2f\n", j*ldc+i, sum);
	    c[j*ldc+i] = alpha*sum;
          }
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

