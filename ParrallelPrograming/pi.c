#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define NUM_POINTS 100000000
#define MILLION 1000000

int points_inside_circle = 0;
pthread_mutex_t lock;
int NUM_THREADS = 1;

void *compute_pi(void *arg)
{
    int thread_id = *((int *)arg);
    int points_in_thread = NUM_POINTS / NUM_THREADS;
    int points_inside_thread = 0;

    for (int i = 0; i < points_in_thread; i++)
    {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;

        if (x * x + y * y <= 1.0)
        {
            points_inside_thread++;
        }
    }
    pthread_mutex_lock(&lock);
    printf("in thread:%d \n", thread_id); 
    points_inside_circle += points_inside_thread;
    pthread_mutex_unlock(&lock);

    return NULL;
}

int main()
{
    double timedif[2];
    for (int j = 0; j < 2; j++)
    {
        if (j == 1)
        {
            NUM_THREADS = 10;
        }
        points_inside_circle = 0;

        pthread_t threads[NUM_THREADS];
        pthread_mutex_init(&lock, NULL);
        // 计时开始
        struct timespec tpstart;
        struct timespec tpend;
        clock_gettime(CLOCK_MONOTONIC, &tpstart);

        int thread_ids[NUM_THREADS]; 
        for (int i = 0; i < NUM_THREADS; i++)
        {
            thread_ids[i] = i;
            pthread_create(&threads[i], NULL, compute_pi, &thread_ids[i]);
        }

        for (int i = 0; i < NUM_THREADS; i++)
        {
            pthread_join(threads[i], NULL);
        }

        pthread_mutex_destroy(&lock);
        clock_gettime(CLOCK_MONOTONIC, &tpend);
        timedif[j] = (tpend.tv_sec - tpstart.tv_sec) + (double)(tpend.tv_nsec - tpstart.tv_nsec) / 1e9;

        double pi_estimate = 4.0 * (double)points_inside_circle / NUM_POINTS;
        if (j == 1)
        {
            printf("Multi-threaded Pi calculation:\n");
        }
        else
        {
            printf("Single-threaded Pi calculation:\n");
        }
        printf("Estimated value of Pi: %lf\n", pi_estimate);
        printf("using %lf s\n", timedif[j]);
    }

    double S8 = timedif[0] / timedif[1];
    double E8 = S8 / 8;
    printf("\nSpeedup: %lf\nParallel Efficiency: %lf\n", S8, E8);
    return 0;
}
