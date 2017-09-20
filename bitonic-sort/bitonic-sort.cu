#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "helper_cuda.h"

int max_power_of_two_less_than(int x)
{
        int result = 1;
        while (result < x) {
                result <<= 1;
        }

        return result >> 1;
}

void compare_swap(int *array, int a, int b, bool up)
{
        if (up == (array[a] > array[b])) {
                int tmp = array[a];
                array[a] = array[b];
                array[b] = tmp;
        }
}

void cpu_bitonic_merge(int *array, int start, int size, bool up)
{
        if (size <= 1)
                return;
        
        int m = max_power_of_two_less_than(size);
        for (int i = start; i < start + size - m; i++) {
                compare_swap(array, i, i + m, up);
        }

        // we have two bitonic subarraies now
        // max of the first one <= min of the second one
        // continue the merge recursively  
        cpu_bitonic_merge(array, start, m, up);
        cpu_bitonic_merge(array, start + m, size - m, up);
}

void cpu_bitonic_sort(int *array, int start, int size, bool up)
{
        if (size <= 1)
                return;

        // sort the first subarray in the reverse direction
        cpu_bitonic_sort(array, start, size / 2, !up);
        // sort the second subarray in the same direction
        cpu_bitonic_sort(array, start + size / 2, size - size / 2, up);

        // merge two sorted subarraies into a sorted one
        cpu_bitonic_merge(array, start, size, up);
}

void cpu_sort(int *array, int size, bool up)
{
        cpu_bitonic_sort(array, 0, size, up);
}

__device__ void swap(int *array, int a, int b)
{
        int tmp = array[a];
        array[a] = array[b];
        array[b] = tmp;
}

__global__ void gpu_bitonic_sort_step(int *d_in, int k, int j, int size)
{
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int ixj = tid ^ j;

        if (tid >= size || ixj <= tid)
                return;
        
        if ((tid & k) == 0) {
                // ascstoping
                if (d_in[tid] > d_in[ixj]) {
                        swap(d_in, tid, ixj);
                } 
        } else {
                // descstoping
                if (d_in[tid] < d_in[ixj]) {
                        swap(d_in, tid, ixj);
                }                 
        }
}

__global__ void gpu_memset(int *d_in, int val, int size)
{
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < size) {
                d_in[tid] = val;
        }
}

void gpu_bitonic_sort(int *h_in, int size)
{
        int *d_in;
        // # of threads in total, # of threads per block, # of blocks in total
        int threads, threads_per_block, blocks;
        // size after padding (must be power of 2)
        int padding_size = max_power_of_two_less_than(size) * 2;

        // allocate GPU memory 
        checkCudaErrors(cudaMalloc((void**)&d_in, sizeof(int) * padding_size));
        // copy input from host memory to GPU memory
        checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(int) * size, cudaMemcpyHostToDevice));
        
        // pad the GPU memory
        if (padding_size > size) {
                threads = padding_size - size;
                threads_per_block = min(1024, threads);
                blocks = (threads + threads_per_block - 1) / threads_per_block;
                gpu_memset<<<blocks, threads_per_block>>>(d_in + size, INT_MAX, padding_size - size);
        }
     
        threads = padding_size;
        threads_per_block = min(1024, threads);
        blocks = (threads + threads_per_block - 1) / threads_per_block;        
        
        // major step
        for (int k = 2; k <= padding_size; k <<= 1) {
                // minor step
                for (int j = k / 2; j > 0; j >>= 1) {
                        gpu_bitonic_sort_step<<<blocks, threads_per_block>>>(d_in, k, j, padding_size);          
                }
        }

        // copy output from GPU memory to host memoory
        cudaMemcpy(h_in, d_in, sizeof(int) * size, cudaMemcpyDeviceToHost);
        // free GPU memory
        cudaFree(d_in);
}

int main()
{
        int array_size = 1234567;
        int *array;
        // sorted result computed by GPU
        int *h_in;
        bool result;
        struct timeval start_time, stop_time;
        double elapsed_time;

        // allocate host memory
        array = (int*)malloc(sizeof(int) * array_size);
        h_in = (int*)malloc(sizeof(int) * array_size);
        if (!array || !h_in) {
                goto out;
        }

        // initialize random number generator
        srand(time(NULL));

        //printf("Input\n");
        for (int i = 0; i < array_size; i++) {
                array[i] = rand() % array_size;
                h_in[i] = array[i];
                //printf("%d ", array[i]);
        }
        //printf("\n");

        // sort on CPU
        gettimeofday(&start_time, NULL);
        cpu_sort(array, array_size, true);
        gettimeofday(&stop_time, NULL);
        elapsed_time = 1000.0 * (stop_time.tv_sec - start_time.tv_sec) + (stop_time.tv_usec - start_time.tv_usec) / 1000.0;
        printf("CPU time: %.3f ms\n", elapsed_time);

        // warm up
        cudaFree(0);

        // sort on GPU
        gettimeofday(&start_time, NULL);
        gpu_bitonic_sort(h_in, array_size);
        gettimeofday(&stop_time, NULL);
        elapsed_time = 1000 * (stop_time.tv_sec - start_time.tv_sec) + (stop_time.tv_usec - start_time.tv_usec) / 1000;
        printf("GPU time: %.3f ms\n", elapsed_time);
        
        // check GPU sort results
        result = true;
        //printf("GPU Output\n");
        for (int i = 0; i < array_size; i++) {
                //printf("%d ", h_in[i]);
                if (h_in[i] != array[i]) {
                        h_in[i] = false;
                }
        }         
        //printf("\n");  

        if (result) {
                printf("Correct\n");
        } else {
                printf("Wrong\n");
        }

out:
        free(array);
        free(h_in);
        return 0;
}