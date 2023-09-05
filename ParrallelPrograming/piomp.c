#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<time.h>

static long num_steps=100000;
double step,pi;

int main()
{
    int i;
    double x,sum =0.0;

    step=1.0/(double)num_steps;
    int max_threads =omp_get_max_threads();
    double start=omp_get_wtime();
#pragma omp parallel for private(i,x) reduction(+:sum)
    for (i=0;i<num_steps;i++){
        x=(i+0.5)*step;
        sum=sum+4.0/(1.0+x*x);
    }
    pi =step*sum;
    double end=omp_get_wtime();
    double time =(end-start);
    printf("pi:%f,using%f",pi,time);
}