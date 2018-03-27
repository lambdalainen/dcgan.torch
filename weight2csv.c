#include <stdio.h>
#include <stdlib.h>

int main()
{
    int nInputPlane = 100;
    int nOutputPlane = 512;
    int kW = 4;
    int kH = 4;

    long weight_size = nInputPlane * nOutputPlane * kW * kH;
    float *weight = calloc(weight_size, sizeof(float));
    float *bias = calloc(nOutputPlane, sizeof(float));

    FILE *fp = fopen("bin/weight_1.bin", "rb");
    fread(weight, sizeof(float), weight_size, fp);
    fclose(fp);
    fp = fopen("bin/bias_1.bin", "rb");
    fread(bias, sizeof(float), nOutputPlane, fp);
    fclose(fp);

    fp = fopen("csv/weight_1.csv", "w");
    for (int i = 0; i < weight_size; i++)
        fprintf(fp, "%f\n", weight[i]);
    fclose(fp);

    fp = fopen("csv/bias_1.csv", "w");
    for (int i = 0; i < nOutputPlane; i++)
        fprintf(fp, "%f\n", bias[i]);
    fclose(fp);

    free(weight);
    free(bias);
    return 0;
}

