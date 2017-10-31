#include <stdio.h>
#include <stdlib.h>
#include <sched.h>

void usage(char *program)
{     
        fprintf(stderr, "usage: %s memsize iters [-a]\n", program);
        fprintf(stderr, "    memsize: memory transferred in bytes (>0)\n");
        fprintf(stderr, "    iters  : number of iterations (>0)\n");
        fprintf(stderr, "    -a     : set CPU affinity\n");
}

// copy size worth of bytes from h_in to d, then from d back to h_out for iters rounds
void profile_memcpy(char *h_in, char *h_out, char *d, int size, int iters)
{
        cudaEvent_t start, stop;
        float time, h2d_total_time, d2h_total_time, throughput, avg_time;
        int i;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        h2d_total_time = d2h_total_time = 0;

        for (i = 0; i < iters; i++) {
                // memory transfer from host to device
                cudaEventRecord(start, 0);
                if (cudaMemcpy(d, h_in, size, cudaMemcpyHostToDevice) != cudaSuccess) {
                        fprintf(stderr, "Error: memcpy from host to device\n");
                }
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time, start, stop);
                h2d_total_time += time;    

                // memory transfer from device to host
                cudaEventRecord(start, 0);
                if (cudaMemcpy(h_out, d, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
                        fprintf(stderr, "Error: memcpy from device to host\n");
                }
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time, start, stop);
                d2h_total_time += time;     
        }

        // calculate host to device information
        avg_time = h2d_total_time / iters / 1000;  // time in second
        throughput = (float)size / avg_time / 1000000000; // throughput in GB/s

        printf("  Host to Device Time: %.6f s\n", avg_time);
        printf("  Host to Device Throughput: %.6f GB/s\n", throughput);

        // calculate device to host information
        avg_time = d2h_total_time / iters / 1000;  // time in second
        throughput = (float)size / avg_time / 1000000000; // throughput in GB/s

        printf("  Device to Host Time: %.6f s\n", avg_time);
        printf("  Device to Host Throughput: %.6f GB/s\n", throughput);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
}

int main(int argc, char **argv)
{
        int size, iters;
        char *h_in_pageable, *h_out_pageable;   // host pageable memory 
        char *h_in_pinned, *h_out_pinned;       // host pinned memory 
        char *d;        // device memory
        bool set_affinity = false;     
        cpu_set_t  mask;

        if (!(argc == 3 || (argc == 4 && strcmp(argv[3], "-a") == 0))) {
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
                set_affinity = true;
        }

        // set a process's CPU affinity mask
        if (set_affinity) {
                CPU_ZERO(&mask);
                CPU_SET(0, &mask);
                if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
                        fprintf(stderr, "Error: set CPU affinity\n");
                        return EXIT_FAILURE;
                }
        }

        // allocate host pageable memory
        h_in_pageable = (char*)malloc(size);
        h_out_pageable = (char*)malloc(size);
        if (!h_in_pageable || !h_out_pageable) {
                fprintf(stderr, "Error: allocate host pageable memory\n");
                free(h_in_pageable);
                free(h_out_pageable);
                return EXIT_FAILURE;
        }

        // allocate host pinned memory
        if (cudaMallocHost((void**)&h_in_pinned, size) != cudaSuccess ||
            cudaMallocHost((void**)&h_out_pinned, size) != cudaSuccess) {
                fprintf(stderr, "Error: allocate host pinned memory\n");
                free(h_in_pageable);
                free(h_out_pageable);
                cudaFreeHost(h_in_pinned);
                cudaFreeHost(h_out_pinned);
                return EXIT_FAILURE;                                    
        } 

        // allocate device memory
        if (cudaMalloc(&d, size) != cudaSuccess) {
                fprintf(stderr, "Error: allocate device memory\n"); 
                free(h_in_pageable);
                free(h_out_pageable);
                cudaFreeHost(h_in_pinned);
                cudaFreeHost(h_out_pinned);   
                return EXIT_FAILURE;
        }

        // warm up
        cudaFree(0);

        // Profile memory copy
        printf("Transfer size (MB): %f\n\n", (float)size / (1024 * 1024));
        printf("Pageable transfers\n");
        profile_memcpy(h_in_pageable, h_out_pageable, d, size, iters);
        printf("\n");
        printf("Pinned transfers\n");
        profile_memcpy(h_in_pinned, h_out_pinned, d, size, iters);

        // free memory
        free(h_in_pageable);
        free(h_out_pageable);
        cudaFreeHost(h_in_pinned);
        cudaFreeHost(h_out_pinned);   
        cudaFree(d);

        return EXIT_SUCCESS;
}