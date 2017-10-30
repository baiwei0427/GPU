#include <stdio.h>
#include <stdlib.h>

void usage(char *program)
{     
        fprintf(stderr, "usage: %s memsize iters\n", program);
        fprintf(stderr, "    memsize: memory transferred in bytes (>0)\n");
        fprintf(stderr, "    iters: number of iterations (>0)\n");
}

int main(int argc, char **argv)
{
        int size, iters, i;
        void *h_ptr, *d_ptr;    // host memory, device memory
        cudaEvent_t start, stop;
        float time, cumulative_h2d_time = 0.f, cumulative_d2h_time = 0.f, throughput, avg_time;

        if (argc != 3) {
                usage(argv[0]);
                return EXIT_FAILURE;
        }

        size = atoi(argv[1]);
        iters = atoi(argv[2]);

        if (size <= 0 || iters <= 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
        }

        // allocate host memory
        if (!(h_ptr = malloc(size))) {
                fprintf(stderr, "Error: allocate host memory\n");
                return EXIT_FAILURE;
        }

        // allocate device memory
        if (cudaMalloc(&d_ptr, size) != cudaSuccess) {
                fprintf(stderr, "Error: allocate device memory\n"); 
                free(h_ptr);              
                return EXIT_FAILURE;
        }

        cudaFree(0);
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (i = 0; i < iters; i++) {
                // memory transfer from host to device
                cudaEventRecord(start, 0);
                if (cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
                        fprintf(stderr, "Error: memcpy from host to device\n");
                }
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time, start, stop);
                cumulative_h2d_time += time;    

                // memory transfer from device to host
                cudaEventRecord(start, 0);
                if (cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
                        fprintf(stderr, "Error: memcpy from device to host\n");
                }
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time, start, stop);
                cumulative_d2h_time += time;                    
        }

        free(h_ptr);
        cudaFree(d_ptr);

        // calculate host to device information
        avg_time = cumulative_h2d_time / iters / 1000;  // time in second
        throughput = size / avg_time / 1000000; // throughput in MB/s
        
        printf("Host to Device Time: %.5f ms\n", avg_time);
        printf("Host to Device Throughput: %.5f MB/s\n", throughput);

        // calculate device to host information
        avg_time = cumulative_d2h_time / iters / 1000;  // time in second
        throughput = size / avg_time / 1000000; // throughput in MB/s
        
        printf("Device to Host Time: %.5f ms\n", avg_time);
        printf("Device to Host Throughput: %.5f MB/s\n", throughput);

        return EXIT_SUCCESS;
}