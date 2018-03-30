#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    FILE *fp = fopen(argv[1], "rb");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    long n_float = size / sizeof(float);
    float *data = calloc(size, sizeof(float));
    rewind(fp);
    fread(data, sizeof(float), n_float, fp);
    fclose(fp);

    char csv_path[256];
    strcpy(csv_path, argv[1]);
    for (char *p = csv_path + strlen(csv_path) - 1; p != csv_path; p--) {
        if (*p == '.') {
            strcpy(p, ".csv");
            break;
        }
    }
    printf("csv_path: %s\n", csv_path);

    fp = fopen(csv_path, "w");
    for (int i = 0; i < n_float; i++)
        fprintf(fp, "%f\n", data[i]);
    fclose(fp);

    free(data);
    return 0;
}

