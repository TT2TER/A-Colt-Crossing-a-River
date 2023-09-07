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

__global__ void add(int *a, int *b, int *c, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
}

__global__ void stencil_1d(int *in, int *out, int n)
{
    __shared__ int temp[THREADS_PER_BLOCK + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;
    // Read input elements into shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS)
    {
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + THREADS_PER_BLOCK] = in[gindex + THREADS_PER_BLOCK];
    }
    // Synchronize (ensure all the data is available)
    __syncthreads();
    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
        result += temp[lindex + offset];
    // Store the result
    out[gindex] = result;
}

int main(void)
{
    int *a, *b, *c;       // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);
    
    // Allocate space for host copies of a, b, c and setup input values
    a = (int *)malloc(size);
    // Initialize 'a' with data
    random_ints(a, N);

    b = (int *)malloc(size);
    // Initialize 'b' with data
    random_ints(b, N);

    c = (int *)malloc(size);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    add<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print part of the result
    for (int i = 0; i < 20; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
