#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void reduce_kernel0(float *d_out, float *d_in)
{
    extern __shared__ float s_data[];

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
float reduce(float *h_in, int array_size)
{
    float result = 0;
    // # of threads per block. It should be the power of two
    int threads = 1 << 10;
    // # of blocks in total. 
    int blocks = 1;
    // GPU memory pointers
    float *d_in, *d_intermediate, *d_out;

    if (!h_in || array_size <= 0 || !is_power_of_2(array_size))
        goto out;

    if (array_size > threads)
        blocks = array_size / threads;
    
    // allocate GPU memory
    if (cudaMalloc((void**) &d_in, array_size * sizeof(float)) != cudaSuccess
     || cudaMalloc((void**) &d_intermediate, blocks * sizeof(float)) != cudaSuccess
     || cudaMalloc((void**) &d_out, sizeof(float)) != cudaSuccess)
        goto out;
    

    //printf("Shared memory per block in bytes: %d\n", threads * sizeof(float));
    // copy the input array from the host memory to the GPU memory
    cudaMemcpy(d_in, h_in, array_size * sizeof(float), cudaMemcpyHostToDevice);
    // first stage reduce
    reduce_kernel0<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
    
    threads = blocks;
    blocks = 1;

    //printf("Shared memory per block in bytes: %d\n", threads * sizeof(float));
    // second stage reduce    
    reduce_kernel0<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_intermediate);
    // copy the result from the GPU memory to the host memory
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);   

out:
    // free GPU memory
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);
    return result;
}

int main() 
{
    const int ARRAY_SIZE = 1 << 20;
    float h_in[ARRAY_SIZE];
    float sum = 0.0f;

    for (int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [-1.0f, 1.0f]
        h_in[i] = -1.0f + (float)random()/((float)RAND_MAX / 2.0f);
        sum += h_in[i];
    }

    printf("Sum: %f\n", sum);
    // use reduce on GPU to calculate the sum
    printf("Reduction Sum: %f\n", reduce(h_in, ARRAY_SIZE));

    return 0;
}