
#include <stdio.h>
#include <math.h>

#define MAX 101
int main()
{
    int data[MAX];
    int trans[MAX][MAX];
    int n = 0, i = 0, k = 0, j = 0;

    while (scanf("%d", &data[n]) != EOF)
        n++;

    int num = 0;

    // 转为矩阵
    for (i = 0; i < sqrt(n); i++)
    {
        for (j = 0; j < sqrt(n); j++)
        {
            trans[i][j] = data[j + num];
        }
        num += (int)sqrt(n);
    }

    // 传递闭包Warshall算法
    for (k = 0; k < sqrt(n); k++)
    {
        for (i = 0; i < sqrt(n); i++)
        {
            for (j = 0; j < sqrt(n); j++)
            {
                trans[i][j] = trans[i][j] || (trans[i][k] && trans[k][j]);
            }
        }
    }

    // 打印结果
    for (i = 0; i < sqrt(n); i++)
    {
        for (j = 0; j < sqrt(n); j++)
        {
            if (j < (int)sqrt(n) - 1)
                printf("%d ", trans[i][j]);
            else
                printf("%d", trans[i][j]);
        }
        printf("\n");
    }
    return 0;
}