#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define WIDTH 800
#define HEIGHT 800
#define MAX_ITER 1000

int mandelbrot(double real, double imag)
{
    int n;
    double r = 0.0;
    double i = 0.0;

    for (n = 0; n < MAX_ITER; n++)
    {
        double r2 = r * r;
        double i2 = i * i;

        if (r2 + i2 > 4.0)
        {
            return n;
        }

        i = 2 * r * i + imag;
        r = r2 - i2 + real;
    }

    return MAX_ITER;
}

int main()
{
    int image[WIDTH][HEIGHT];

#pragma omp parallel for collapse(2)
    for (int x = 0; x < WIDTH; x++)
    {
        for (int y = 0; y < HEIGHT; y++)
        {
            double real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            double imag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;

            int value = mandelbrot(real, imag);

#pragma omp critical
            {
                image[x][y] = value;
            }
        }
    }

    // 输出PGM图像文件
    FILE *fp = fopen("mandelbrot.pgm", "wb");
    fprintf(fp, "P2\n");
    fprintf(fp, "%d %d\n", WIDTH, HEIGHT);
    fprintf(fp, "255\n");

    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            fprintf(fp, "%d ", image[x][y]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    return 0;
}
