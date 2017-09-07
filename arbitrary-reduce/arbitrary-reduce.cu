#include <stdio.h>
#include <stdlib.h>

#include "helper_cuda.h"

__global__ void reduce_max_kernel(const int *d_in, unsigned int in_size, int *d_out)
{
    extern __shared__ int s_data[];
    
    // thread ID inside the block
    unsigned int tid = threadIdx.x;
    // global ID across all blocks
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // # of elements in this block
    unsigned int block_size = blockDim.x;

    // if it is the last block
    if (blockIdx.x == gridDim.x - 1) {
      block_size -= gridDim.x * blockDim.x - in_size;
    }

    // copy elements from global memoery into per-block shared memory
    if (tid < block_size) {
      s_data[tid] = d_in[gid];
    } 
    __syncthreads();

	// ceil(block_size / 2.0)
    unsigned int s = (block_size + 1) << 1;

    while (s > 0) {
      if (tid < s && tid + s < block_size) {
        s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
      }

	  s = min((s + 1) << 1, s - 1);
      __syncthreads();
    }

    // write output from shared memory to global memory
    if (tid == 0) {
        d_out[blockIdx.x] = s_data[0];
    }
}

int reduce_max(int *h_in, unsigned int size)
{
	// GPU memory
	int *d_in, *d_inter, *d_out;
	// host output
	int h_out;
	// # of blocks, # of threads per block
	unsigned int blocks, threads_per_block;
	// size of d_inter
	unsigned int inter_size;	

	threads_per_block = 1024;
	blocks = (size + threads_per_block - 1) / threads_per_block;
	inter_size = blocks;

	// allocate GPU memory
	checkCudaErrors(cudaMalloc((void**)&d_in, size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_inter, inter_size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_out, sizeof(int)));

	// copy input from host memory to GPU memory
	cudaMemcpy(d_in, h_in, size * sizeof(int), cudaMemcpyHostToDevice);

	// launch kernels;
	reduce_max_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(int)>>>(d_in, size, d_inter);
	reduce_max_kernel<<<1, blocks, blocks * sizeof(int)>>>(d_inter, inter_size, d_out);

	// copy output from GPU memory to host memory
	cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

	// free GPU memory
	checkCudaErrors(cudaFree(d_in));
	checkCudaErrors(cudaFree(d_inter));
	checkCudaErrors(cudaFree(d_out));

	return h_out;
}

int main(int argc, char **argv) 
{
    const int ARRAY_SIZE = 20000;
    int h_in[ARRAY_SIZE];
	int result, expect_result = INT_MIN;

    // initialize random number generator	
	srand(time(NULL));
    
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = rand() / 100;
		expect_result = fmaxf(expect_result, h_in[i]); 
    }

	printf("The maximum number is %d\n", expect_result);

	result = reduce_max(h_in, ARRAY_SIZE);
	printf("The output of reduce_max() is %d\n", result);	
	
	if (result == expect_result) {
		printf("Correct\n");
	} else {
		printf("Wrong\n");
	}

    return 0;
}