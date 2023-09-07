%%cuda --name testGoogleColab.cu
#include <iostream>
#include <cstdlib>

#define N (2048 * 2048)
#define THREADS_PER_BLOCK 512
#define RADIUS 2 // Define the stencil radius

// Function to generate random integers and fill the array
__host__ void random_ints(int* array, int size) {
    // Set the seed for the random number generator
    srand(time(NULL));

    for (int i = 0; i < size; i++) {
        array[i] = rand() % 100; // Generate random integers between 0 and 99 (adjust as needed)
    }
}

__global__ void stencil_1d(int *in, int *out, int n)
{
    __shared__ int temp[THREADS_PER_BLOCK + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;
    temp[lindex] = in[gindex];

    // Check if threadIdx.x is less than RADIUS
    if (threadIdx.x < RADIUS)
    {
        // Ensure that the indices are within bounds before accessing the array
        if (lindex - RADIUS >= 0)
            temp[lindex - RADIUS] = in[gindex - RADIUS];
        if (lindex + THREADS_PER_BLOCK < n)
            temp[lindex + THREADS_PER_BLOCK] = in[gindex + THREADS_PER_BLOCK];
    }
    __syncthreads();

    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
    {
        int neighborIndex = lindex + offset;
        if (neighborIndex >= 0 && neighborIndex < THREADS_PER_BLOCK + 2 * RADIUS)
            result += temp[neighborIndex];
    }

    // Store the result
    out[gindex] = result;
}


int main(void)
{
    int *a, *b;       // host copies of a, b, c
    int *d_a, *d_b; // device copies of a, b, c
    int size = N * sizeof(int);
    
    // Allocate space for host copies of a, b, c and setup input values
    a = (int *)malloc(size);
    random_ints(a, N);

    b = (int *)malloc(size);

    cudaMalloc((void **)&d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_b, size);


    // Launch add() kernel on GPU
    stencil_1d<<<N  / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, N);

    // Copy result back to host
    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

    // Print part of the result
    std::cout<<"first 30 origin data is:"<<std::endl;
    for (int i = 0; i < 30; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    std::cout<<"when radius is "<<RADIUS<<"first 30 stenciled data is:"<<std::endl;
    for (int i = 0; i < 30; i++) {
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;

    std::cout<<"last 30 origin data is:"<<std::endl;
    for (int i = N-30; i < N; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    std::cout<<"when radius is "<<RADIUS<<" last 30 stenciled data is:"<<std::endl;
    for (int i = N-30; i < N; i++) {
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;

    srand(time(NULL));
    int middle =(rand()%100)*(rand()%100);
    std::cout<<"from "<<middle<<" middle 30 origin data is:"<<std::endl;
    for (int i = middle-30; i < middle; i++) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;

    std::cout<<"when radius is "<<RADIUS<<" middle 30 stenciled data is:"<<std::endl;
    for (int i = middle-30; i < middle; i++) {
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    free(a);
    free(b);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}