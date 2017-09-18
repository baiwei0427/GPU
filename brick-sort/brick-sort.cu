#include <stdio.h>
#include "helper_cuda.h"

inline void swap(int *array, unsigned int i, unsigned int j)
{
        int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
}

void brick_sort(int *array, unsigned int size)
{
        bool sorted = false;
        while (!sorted) {
                sorted = true;
                // odd sort
                for (unsigned int i = 1; i < size - 1; i += 2) {
                        if (array[i] > array[i + 1]) {
                                swap(array, i, i + 1);
                                sorted = false;
                        }
                }
                // even sort
                for (unsigned int i = 0; i < size - 1; i += 2) {
                        if (array[i] > array[i + 1]) {
                                swap(array, i, i + 1);
                                sorted = false;
                        }
                }
        }
}

__global__ void brick_sort_kernel(int *h_in, bool is_even, unsigned int size)
{
        // global thread ID
        unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
        int tmp;

        // even phase
        if (is_even && gid * 2 + 1 < size && h_in[gid * 2] > h_in[gid * 2 + 1]) {
                tmp = h_in[gid * 2];
                h_in[gid * 2] = h_in[gid * 2 + 1];
                h_in[gid * 2 + 1] = tmp;
        
        // odd phase
        } else if (!is_even && gid * 2 + 2 < size && h_in[gid * 2 + 1] > h_in[gid * 2 + 2]) {
                tmp = h_in[gid * 2 + 1];
                h_in[gid * 2 + 1] = h_in[gid * 2 + 2];
                h_in[gid * 2 + 2] = tmp;                
        }
}

// brick sort on GPU
void gpu_brick_sort(int *h_in, unsigned int size)
{
        int *d_in;
        unsigned int blocks, threads_per_block;

        // no need to sort
        if (size == 1) {
                return;
        }

	// allocate GPU memory
	checkCudaErrors(cudaMalloc((void**)&d_in, size * sizeof(int)));
        // copy input from host memory to GPU memory
        cudaMemcpy(d_in, h_in, size * sizeof(int), cudaMemcpyHostToDevice);
        
        // lauch kernels to do computation
        // launch size / 2 threads in total
        // run size - 1 rounds
        threads_per_block = 1024;
        blocks = (size / 2 + threads_per_block - 1) / threads_per_block;
        if (blocks == 1) {
                threads_per_block = min(size, threads_per_block);
        }

        for (int i = 0; i < size - 1; i++) {
                brick_sort_kernel<<<blocks, threads_per_block>>>(d_in, i % 2, size);
        }

        // copy output from GPU memory to host memory
        cudaMemcpy(h_in, d_in, size * sizeof(int), cudaMemcpyDeviceToHost);
        // free GPU memory
        cudaFree(d_in);        
}

int main() 
{       
        const int array_size = 1000;
        int array[array_size];
        // sort result computed by GPU
        int h_in[array_size];
        bool result;

        // initialize random number generator
        srand(time(NULL));

        // generate input
        printf("Input:\n");
        for (int i = 0; i < array_size; i++) {
                array[i] = rand() % array_size;
                h_in[i] = array[i];
                printf("%d ", array[i]);
        }
        printf("\n");

        // brick sort on GPU
        gpu_brick_sort(h_in, array_size);
        printf("GPU Output:\n");
        for (int i = 0; i < array_size; i++) {
                printf("%d ", h_in[i]);
        }        
        printf("\n");      

        // brick sort on CPU
        result = true;
        brick_sort(array, array_size);
        printf("Expected Output:\n");
        for (int i = 0; i < array_size; i++) {
                printf("%d ", array[i]);
                if (array[i] != h_in[i]) {
                        result = false;
                }
        }        
        printf("\n");

        if (result) {
                printf("Correct result\n");
        } else {
                printf("Wrong result\n");
        }

        return 0;
}