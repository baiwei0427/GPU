#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include "helper_cuda.h"

__global__ void histo_kernel2(unsigned char *d_in, unsigned int d_in_size, unsigned int *d_out)
{
        unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
        // total # of launched threads 
        unsigned int stride = blockDim.x * gridDim.x;
        
        while (i < d_in_size) {
                atomicAdd(&d_out[d_in[i]], 1);
                i += stride;     
        }   
}

// do histogram computation on GPU
// input: h_in (size: h_in_size), output: h_out (size: h_out_size)
void gpu_histo2(unsigned char *h_in, unsigned int h_in_size, unsigned int *h_out, unsigned int h_out_size, unsigned int iters)
{
        // input on GPU memory
        unsigned char *d_in;
        // output on GPU memory
        unsigned int *d_out;
        unsigned int grid, block;
        cudaDeviceProp prop;

        if (iters == 0 || h_in_size == 0 || h_out_size == 0)
                return; 

        // allocate GPU memory
        if (cudaMalloc((void**) &d_in, h_in_size* sizeof(unsigned char)) != cudaSuccess
         || cudaMalloc((void**) &d_out, h_out_size * sizeof(unsigned int)) != cudaSuccess)
                goto out;
        
        // copy input from the host memory to GPU memory
        cudaMemcpy(d_in, h_in, h_in_size * sizeof(unsigned char), cudaMemcpyHostToDevice);        

        checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
        grid = 2 * prop.multiProcessorCount;
        block = h_out_size;

        // for each round
        for (int i = 0; i < iters; i++) {
                // initialize d_out to all 0
                cudaMemset(d_out, 0, h_out_size * sizeof(unsigned int));
                // launch kernel
                histo_kernel2<<<grid, block>>>(d_in, h_in_size, d_out);
        }

        // copy the result from the GPU memory to the host memory
        cudaMemcpy(h_out, d_out, h_out_size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

out:
        // free GPU memory
        cudaFree(d_in);
        cudaFree(d_out); 
}

__global__ void histo_kernel(unsigned char *d_in, unsigned int *d_out)
{
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int id = y * blockDim.x * gridDim.x + x;
        
        atomicAdd(&d_out[d_in[id]], 1);        
}

// do histogram computation on GPU
// input: h_in (size: h_in_size), output: h_out (size: h_out_size)
void gpu_histo(unsigned char *h_in, unsigned int h_in_size, unsigned int *h_out, unsigned int h_out_size, unsigned int iters)
{
        // input on GPU memory
        unsigned char *d_in;
        // output on GPU memory
        unsigned int *d_out;
        dim3 block(32, 32);
        dim3 grid(0, 0);
        unsigned int x, y;

        if (iters == 0 || h_in_size == 0 || h_out_size == 0)
                return; 

        // allocate GPU memory
        if (cudaMalloc((void**) &d_in, h_in_size* sizeof(unsigned char)) != cudaSuccess
         || cudaMalloc((void**) &d_out, h_out_size * sizeof(unsigned int)) != cudaSuccess)
                goto out;
        
        // copy input from the host memory to GPU memory
        cudaMemcpy(d_in, h_in, h_in_size * sizeof(unsigned char), cudaMemcpyHostToDevice);        

        // calculate block and grid sizes
        // we assume that h_in_size > 1024 and h_in_size is a square root of a number
        x = (unsigned int)sqrt(h_in_size);
        y = h_in_size / x;
        grid.x = x / block.x;
        grid.y = y / block.y;

        // for each round
        for (int i = 0; i < iters; i++) {
                // initialize d_out to all 0
                cudaMemset(d_out, 0, h_out_size * sizeof(unsigned int));
                // launch kernel
                histo_kernel<<<grid, block>>>(d_in, d_out);
        }

        // copy the result from the GPU memory to the host memory
        cudaMemcpy(h_out, d_out, h_out_size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

out:
        // free GPU memory
        cudaFree(d_in);
        cudaFree(d_out); 
}

// do histogram computation on array 'buf' for 'iters' times 
// save results to 'histo'
void cpu_histo(unsigned char *buf, unsigned int buf_size, unsigned int *histo, unsigned int histo_size, unsigned int iters)
{
        if (iters == 0 || buf_size == 0 || histo_size == 0)
                return;

        // for each round
        for (int i = 0; i < iters; i++) {
                // initialize elements of histo to all 0
                memset(histo, 0, histo_size * sizeof(int));
                // histogram computation
                for (int j = 0; j < buf_size; j++) {
                        histo[buf[j]]++;
                }
        }
}

int main(int argc, char **argv)
{
        // size of histogram
        unsigned int histo_size = 256;
        // GPU computation results
        unsigned int histo[histo_size];
        // expected results (by CPU)
        unsigned int expected_histo[histo_size];

        // size of array: 100MB
        unsigned int array_size = 100 << 20;
        unsigned char *array = (unsigned char*)malloc(array_size);

        unsigned int iters = 1;
        struct timeval start_time, stop_time;
        double elapsed_time;

        if (!array) {
                exit(EXIT_FAILURE);
        }

        srand((unsigned int)time(NULL));
        
        // initialize array with random numbers
        for (int i = 0; i < array_size; i++) {
                // generate a random number in range [0, 255]
                array[i] = rand() & 0xff;
        }

        gettimeofday(&start_time, NULL);
        // calculate histogram results on CPU
        cpu_histo(array, array_size, expected_histo, histo_size, 1);
        gettimeofday(&stop_time, NULL);
        elapsed_time = (stop_time.tv_sec - start_time.tv_sec) * 1000 + (stop_time.tv_usec - start_time.tv_usec) / 1000.0;

        printf("CPU computation time: %f ms\n", elapsed_time);

        if (iters == 0) {
                goto out;
        }

        gettimeofday(&start_time, NULL);
        // calculate histogram results on GPU
        gpu_histo(array, array_size, histo, histo_size, iters);
        gettimeofday(&stop_time, NULL);
        elapsed_time = (stop_time.tv_sec - start_time.tv_sec) * 1000 + (stop_time.tv_usec - start_time.tv_usec) / 1000.0;
        
        printf("GPU implementation 1 computation time: %f ms\n", elapsed_time / iters);

        // check results
        for (int i = 0; i < histo_size; i++) {
                //printf("%d %u %u\n", i, histo[i], expected_histo[i]);
                if (histo[i] != expected_histo[i]) {
                        printf("Wrong results\n");
                        goto out;
                }
        }

        printf("Correct results\n");

        gettimeofday(&start_time, NULL);
        // calculate histogram results on GPU
        gpu_histo2(array, array_size, histo, histo_size, iters);
        gettimeofday(&stop_time, NULL);
        elapsed_time = (stop_time.tv_sec - start_time.tv_sec) * 1000 + (stop_time.tv_usec - start_time.tv_usec) / 1000.0;
        
        printf("GPU implementation 2 computation time: %f ms\n", elapsed_time / iters);

        // check results
        for (int i = 0; i < histo_size; i++) {
                //printf("%d %u %u\n", i, histo[i], expected_histo[i]);
                if (histo[i] != expected_histo[i]) {
                        printf("Wrong results\n");
                        goto out;
                }
        }

        printf("Correct results\n");

out:
        free(array);
        return 0;
}