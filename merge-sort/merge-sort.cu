#include <stdio.h>
#include "helper_cuda.h"

void merge(int *array, int start, int mid, int end)
{       
        int left_index, right_index, global_index;
        int left_len = mid - start + 1;
        int right_len = end - mid;
        int left[left_len];
        int right[right_len];

        // initialize left array
        for (int i = 0; i < left_len; i++) {
                left[i] = array[start + i];
        }

        // initialize right array
        for (int i = 0; i < right_len; i++) {
                right[i] = array[mid + 1 + i];
        }

        // index of left array
        left_index = 0;
        // index of right array
        right_index = 0;
        // index of merged array
        global_index = start;

        while (left_index < left_len && right_index < right_len) {
                if (left[left_index] <= right[right_index]) {
                        array[global_index++] = left[left_index++];
                } else {
                        array[global_index++] = right[right_index++];                        
                }
        }

        // copy the rest of left array 
        while (left_index < left_len) {
                array[global_index++] = left[left_index++];
        } 

        // copy the rest of right array
        while (right_index < right_len) {
                array[global_index++] = right[right_index++];
        }
}

void cpu_merge_sort(int *array, int start, int end)
{
        if (start >= end)
                return;
        
        int mid = start + (end - start) / 2;
        cpu_merge_sort(array, start, mid);
        cpu_merge_sort(array, mid + 1, end);
        merge(array, start, mid, end);
}

__global__ void gpu_merge(int *d_in, int *d_out, int size, int sorted_size) 
{
        // global ID
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        // start, end of left subarray
        int left_start = gid * 2 * sorted_size;
        int left_end = min((gid * 2 + 1) * sorted_size - 1, size - 1);
        // start, end of right subarray
        int right_start = (gid * 2 + 1) * sorted_size;
        int right_end = min((gid * 2 + 2) * sorted_size - 1, size - 1);
        
        int left_index = left_start, right_index= right_start, global_index= left_start;

        while (left_index <= left_end && right_index <= right_end) {
                if (d_in[left_index] <= d_in[right_index]) {
                        d_out[global_index++] = d_in[left_index++];
                } else {
                        d_out[global_index++] = d_in[right_index++];                        
                }
        }

        while (left_index <= left_end) {
                d_out[global_index++] = d_in[left_index++];                
        }

        while (right_index <= right_end) {
                d_out[global_index++] = d_in[right_index++];                  
        }
}

void gpu_merge_sort(int *h_in, int size)
{
        int *d_in, *d_out, *tmp;

        if (size == 1)
                return;
        
	// allocate GPU memory
	checkCudaErrors(cudaMalloc((void**)&d_in, size * sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&d_out, size * sizeof(int)));

        // copy input from host memory to GPU memory
        cudaMemcpy(d_in, h_in, size * sizeof(int), cudaMemcpyHostToDevice);

        // # of threads per block
        int threads_per_block = 1024;
        int sorted_size = 1;
        // # of blocks 
        int blocks = 1;

        while (sorted_size < size) {
                // each thread can merge at most 2 * sorted_size elements
                // how many threads do we need in total?
                int threads_total = (size + 2 * sorted_size - 1) / (2 * sorted_size);
                // total # of blocks that we need
                blocks = (threads_total + threads_per_block - 1) / threads_per_block;

                gpu_merge<<<blocks, threads_per_block>>>(d_in, d_out, size, sorted_size);
                
                sorted_size *= 2;
                // exchange input and output
                tmp = d_in;
                d_in = d_out;
                d_out = tmp;
        }

        // copy output from GPU memory to host memory
        cudaMemcpy(h_in, d_in, size * sizeof(int), cudaMemcpyDeviceToHost);

        // free GPU memory
        cudaFree(d_in);
        cudaFree(d_out);
}

int main()
{
        int array_size = 1111;
        int array[array_size];
        // sort result computed by GPU
        int h_in[array_size];
        bool result;

        // initialize random number generator
        srand(time(NULL));

        printf("Input\n");
        for (int i = 0; i < array_size; i++) {
                array[i] = rand() % array_size;
                h_in[i] = array[i];
                printf("%d ", array[i]);
        }
        printf("\n");

        // merge sort on CPU
        cpu_merge_sort(array, 0, array_size - 1);
        printf("Expected Output\n");
        for (int i = 0; i < array_size; i++) {
                printf("%d ", array[i]);
        }         
        printf("\n");        
        
        // merge sort on GPU
        gpu_merge_sort(h_in, array_size);
        printf("GPU Output\n"); 
        result = true;
        for (int i = 0; i < array_size; i++) {
                printf("%d ", h_in[i]);
                if (h_in[i] != array[i]) {
                        result = false;
                }
        }        
        printf("\n");

        if (result) {
                printf("Correct\n");
        } else {
                printf("Wrong\n");
        }
        
        return 0;
}