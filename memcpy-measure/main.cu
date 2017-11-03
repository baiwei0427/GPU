#include <stdio.h>
#include <stdlib.h>

#define HOST_TO_DEVICE 0
#define DEVICE_TO_HOST 1

// Print the usage of the program
inline void usage(char *program)
{     
        fprintf(stderr, "usage: %s memsize iters [-r]\n", program);
        fprintf(stderr, "    memsize : memory transferred in bytes (>0)\n");
        fprintf(stderr, "    iters   : number of iterations (>0)\n");
        fprintf(stderr, "    -r      : re-allocate memory for each iteration\n");
}

// Allocate 'size' worth of bytes to host memory '*ptr'
inline void alloc_host_mem(void **ptr, int size, bool pinned)
{
        if (pinned) {
                cudaMallocHost(ptr, size);
        } else {
                *ptr = malloc(size);
        }
}

// Free host memory pointed by 'ptr'
inline void free_host_mem(void *ptr, bool pinned)
{
        if (pinned) {
                cudaFreeHost(ptr);
        } else {
                free(ptr);
        }
}

// Profile memory copy performance between GPU and CPU
// For each iteration, we copy 'size' worth of bytes in a direction 
// size        :   memory copy size for each iteration
// iters       :   number of iterations
// direction   :   HOST_TO_DEVICE / DEVICE_TO_HOST
// pinned      :   whether enable pinned memory allocation at host
// reallocate  :   whether re-allocate memory for each time
// This function prints the average throughput and transfer time result    
void profile_memcpy(int size, int iters, int direction, bool pinned, bool reallocate)
{
        void *h = NULL, *d = NULL;      // host and device memory
        cudaEvent_t start, stop;
        float time, avg_time, total_time, throughput;
        int i;
        cudaError_t result;

        // Allocate host and device memory only one time
        if (!reallocate) {
                alloc_host_mem(&h, size, pinned);
                cudaMalloc(&d, size);
        }

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        total_time = 0;

        for (i = 0; i < iters; i++) {
                // Re-allocate host and device memory for each time
                if (reallocate) {                        
                        alloc_host_mem(&h, size, pinned);
                        cudaMalloc((void**)&d, size);               
                }

                cudaEventRecord(start, 0);
                if (direction == HOST_TO_DEVICE) { 
                        result = cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
                } else {
                        result = cudaMemcpy(h, d, size, cudaMemcpyDeviceToHost);
                }
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time, start, stop);
                total_time += time;
                
                if (result != cudaSuccess) {
                        fprintf(stderr, "Error: memory copy\n");
                }

                if (reallocate) {
                        free_host_mem(h, pinned);
                        cudaFree(d);
                }
        }

        // Calculate host to device information
        avg_time = total_time / iters / 1000;  // time in second
        throughput = (float)size / avg_time / 1000000000; // throughput in GB/s

        if (direction == HOST_TO_DEVICE) {
                printf("  Host to Device Time: %.6f s\n", avg_time);
                printf("  Host to Device Throughput: %.6f GB/s\n", throughput);
        } else {
                printf("  Device to Host Time: %.6f s\n", avg_time);
                printf("  Device to Host Throughput: %.6f GB/s\n", throughput);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (!reallocate) {
                free_host_mem(h, pinned);
                cudaFree(d);
        }
}

int main(int argc, char **argv)
{
        int size, iters;
        bool reallocate = false;        // By default, we don't re-allocate memory for each test

        if (!(argc == 3 || (argc == 4 && strcmp(argv[3], "-r") == 0))) {
                usage(argv[0]);
                return EXIT_FAILURE;
        }

        size = atoi(argv[1]);
        iters = atoi(argv[2]);

        if (size <= 0 || iters <= 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
        }

        if (argc == 4) {
                reallocate = true;
        }

        // warm up
        cudaFree(0);

        // Profile memory copy
        printf("Transfer size (MB): %f\n\n", (float)size / (1024 * 1024));
        
        printf("Pageable transfers\n");
        profile_memcpy(size, iters, HOST_TO_DEVICE, false, reallocate);
        profile_memcpy(size, iters, DEVICE_TO_HOST, false, reallocate);

        printf("\n");
        
        printf("Pinned transfers\n");
        profile_memcpy(size, iters, HOST_TO_DEVICE, true, reallocate);
        profile_memcpy(size, iters, DEVICE_TO_HOST, true, reallocate);

        return EXIT_SUCCESS;
}