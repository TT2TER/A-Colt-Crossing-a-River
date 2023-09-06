#include <stdio.h>
#include <iostream>
#include <mpi.h>
#include <math.h>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n; // Number of segments
    double mypi = 0.0;

    if (rank == 0) {
        // Read the number of segments from the command line
        // and broadcast it to all processes
        // This way, all processes will know the value of 'n'
        // For example, set n to 10000
        std::cout<<"请输入期望划分精度，推荐值10000"<<std::endl;
        std::cin>>n;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double w = 1.0 / (double)n;
    double sum = 0.0;

    for (int i = rank + 1; i <= n; i += size) {
        double x = w * ((double)i - 0.5);
        sum += sqrt(1.0 - x * x);
    }

    double pi;


    int my_rank, numprocs;
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    int  namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name,&namelen);
    printf("Working! Process %d of %d on %s\n",my_rank,numprocs,processor_name);


    MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi *= 4.0 * w;
        printf("Approximated pi: %.16f, Error: %.16f\n", pi, fabs(pi - M_PI));
    }

    MPI_Finalize();
    return 0;
}
