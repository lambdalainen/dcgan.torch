#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"

static void free_q(struct Q *q, int free_f)
{
    if (free_f)
        free(q->f);
    free(q->q);
    free(q);
}

static struct Q *quantize(float *data, long count)
{
    struct Q *q = malloc(sizeof(struct Q));
    q->f = data;
    q->q = malloc(sizeof(uint8_t) * count);

    float min = FLT_MAX;
    float max = FLT_MIN;
    double scale;
    double initial_zero_point;
    uint8_t zero_point;

    for (long i = 0; i < count; i++) {
        if (data[i] < min)
            min = data[i];
        else if (data[i] > max)
            max = data[i];
    }
    scale = (max - min) / 255;
    initial_zero_point = - min / scale;
    if (initial_zero_point < 0)
        zero_point = 0;
    else if (initial_zero_point > 255)
        zero_point = 255;
    else
        zero_point = round(initial_zero_point);

    q->min = min;
    q->max = max;
    q->s = scale;
    q->z = zero_point;

    for (long i = 0; i < count; i++) {
        float transformed_val = zero_point + data[i] / scale;
        if (transformed_val < 0.0f)
            transformed_val = 0.0f;
        else if (transformed_val > 255.0f)
            transformed_val = 255.0f;
        q->q[i] = round(transformed_val);
    }
    return q;
}

int main(void)
{
    long m = 2;
    long k = 4;
    long n = 3;

    float lhs[8] =  {0.629, 0.812, -0.746,  0.827,
                    -0.729, 0.67,   0.938, -0.558};
    float rhs[12] = {0.265, -0.443,  0.915,
                    -0.384, -0.623,  0.993,
                    -0.805,  0.0938, 0.93,
                     0.0944, 0.986,  0.935};

    printf("lhs 2x4:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++)
            printf("%.3f\t", lhs[i*k+j]);
        printf("\n");
    }

    printf("rhs 4x3:\n");
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++)
            printf("%.3f\t", rhs[i*n+j]);
        printf("\n");
    }

    float lhs_transposed[8];

    printf("\nlhs_transposed 4x2:\n");
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            lhs_transposed[i*m+j] = lhs[j*k+i];
            printf("%.3f\t", lhs_transposed[i*m+j]);
        }
        printf("\n");
    }

    float result[6];

    gemm(
        'n', 't',
        m, n, k,
        1,
        lhs_transposed, m,
        rhs, n,
        0,
        result, m
    );

    printf("result (2x3 but stored in column-major order):\n");
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++)
            printf("%.3f\t", result[j*m+i]);
        printf("\n");
    }

    printf("\nNote that gemm expects column-major, therefore lhs_transposed\n"
           "is passed in, but with ('n', 't', ...), rhs is passed in.\n"
           "The result is in column-major as well\n");

    printf("\nNow, find quantization parameters:\n");
    struct Q *a = quantize(lhs, 8);
    struct Q *b = quantize(rhs, 12);
    struct Q *c = quantize(result, 6);
    printf("lhs: min %.3f max %.3f scale %f zero_point %i\n", a->min, a->max, a->s, a->z);
    printf("rhs: min %.3f max %.3f scale %f zero_point %i\n", b->min, b->max, b->s, b->z);
    printf("result: min %.3f max %.3f scale %f zero_point %i\n", c->min, c->max, c->s, c->z);

    printf("\nQuantized lhs 2x4:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++)
            printf("%i\t", a->q[i*k+j]);
        printf("\n");
    }

    printf("\nQuantized rhs 4x3:\n");
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++)
            printf("%i\t", b->q[i*n+j]);
        printf("\n");
    }

    uint8_t lhs_q_transposed[8];

    printf("\nQuantized lhs_transposed 4x2:\n");
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            lhs_q_transposed[i*m+j] = a->q[j*k+i];
            printf("%i\t", lhs_q_transposed[i*m+j]);
        }
        printf("\n");
    }

    // int32_t to accommodate the accumulating result, but will be
    // scaled down to 0-255 eventually
    int32_t result_fixed[6];

    gemm_fixed(
        'n', 't',
        m, n, k,
        1,
        lhs_q_transposed, m,
        b->q, n,
        0,
        result_fixed, m,
        a, b, c
    );

    // scale down to uint8_t
    float scale = a->s * b->s / c->s;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++) {
            int idx = j*m+i;
            int32_t result = round(result_fixed[idx] * scale) + c->z;
            if (result < 0)
                result = 0;
            else if (result > 255)
                result = 255;
            result_fixed[idx] = result;
        }
    }

    printf("\nResult by gemm_fixed (2x3 but stored in column-major order):\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            printf("%i\t", result_fixed[j*m+i]);
        printf("\n");
    }

    printf("\nDequantized floats from uint8_t:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            printf("%.3f\t", c->s * (result_fixed[j*m+i] - c->z));
        printf("\n");
    }

    free_q(a, 0);
    free_q(b, 0);
    free_q(c, 0);

    return 0;
}

