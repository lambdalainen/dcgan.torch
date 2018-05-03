#include <stdio.h>

int main(void)
{
    int a[8] = {1, 2, 3, 4, 1, 2, 3, 4}; // 2x4
    int b[8]; // 4x2
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            b[j * 2 + i] = a[i * 4 + j];
    for (int i = 0; i < 8; i++)
        printf("%i ", b[i]);
    printf("\n");

    return 0;
}
