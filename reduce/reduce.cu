#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

__global__ void reduce_kernel0(int *d_out, int *d_in)
{
    extern __shared__ int s_data[];

    // thread ID inside the block
    unsigned int tid = threadIdx.x;
    // global ID across all blocks
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy elements from global memoery into per-block shared memory
    s_data[tid] = d_in[gid];
    // Ensure all elements have been copied into shared memory
    __syncthreads();

    // s = 1, 2, 4, 8, ..... blockDim.x / 2
    for (unsigned int s = 1; s < blockDim.x; s = (s << 1)) {
        if (tid % (s << 1) == 0) {
            s_data[tid] += s_data[tid + s];
        }
        // Ensure all threads in the block finish add in this round
        __syncthreads();
    }

    // write the reduction sum back to the global memory
    if (tid == 0) {
        d_out[blockIdx.x] = s_data[0];
    }
}

inline bool is_power_of_2(int n)
{
    return ((n & (n - 1)) == 0);
}


// input: array (in host memory) and array size 
int reduce(int *h_in, int array_size)
{
    int result = 0;
    // # of threads per block. It should be the power of two
    int threads = 1 << 10;
    // # of blocks in total. 
    int blocks = 1;
    // GPU memory pointers
    int *d_in, *d_intermediate, *d_out;

    if (!h_in || array_size <= 0 || !is_power_of_2(array_size))
        goto out;

    if (array_size > threads)
        blocks = array_size / threads;
    
    // allocate GPU memory
    if (cudaMalloc((void**) &d_in, array_size * sizeof(int)) != cudaSuccess
     || cudaMalloc((void**) &d_intermediate, blocks * sizeof(int)) != cudaSuccess
     || cudaMalloc((void**) &d_out, sizeof(int)) != cudaSuccess)
        goto out;
    

    //printf("Shared memory per block in bytes: %d\n", threads * sizeof(int));
    // copy the input array from the host memory to the GPU memory
    cudaMemcpy(d_in, h_in, array_size * sizeof(int), cudaMemcpyHostToDevice);
    // first stage reduce
    reduce_kernel0<<<blocks, threads, threads * sizeof(int)>>>(d_intermediate, d_in);
    
    threads = blocks;
    blocks = 1;

    //printf("Shared memory per block in bytes: %d\n", threads * sizeof(int));
    // second stage reduce    
    reduce_kernel0<<<blocks, threads, threads * sizeof(int)>>>(d_out, d_intermediate);
    // copy the result from the GPU memory to the host memory
    cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost);   

out:
    // free GPU memory
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);
    return result;
}

// generate a random integer in [min, max]
inline int random_range(int min, int max)
{
    if (min > max)
        return 0;
    else
        return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

int main() 
{
    const int ARRAY_SIZE = 1 << 20;
    int h_in[ARRAY_SIZE];
    int sum = 0;
    
    // initialize random number generator
    srand(time(NULL));
    int min = 0, max = 10;

    for (int i = 0; i < ARRAY_SIZE; i++) {
        // generate a random int in a range
        h_in[i] = random_range(min, max);
        sum += h_in[i];
    }

    const int iters = 50;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; i++) {
        // wrong result
        int result = reduce(h_in, ARRAY_SIZE);
        if (result != sum) {
            printf("Wrong result: %d and %d\n", sum, result);
        }
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);    
    elapsed_time /= iters;      

    printf("Average time elapsed: %f ms\n", elapsed_time);

    return 0;
}