#include <stdio.h>
#include <stdlib.h>

__global__ void SyncKernel(int iters) 
{       
        int i;

        for (i = 0; i < iters; i++) {
                __syncthreads();
        }
}

void usage(char *program)
{       
        fprintf(stderr, "usage: %s nblocks nthreads iters\n", program);
        fprintf(stderr, "    nblocks: number of thread blocks (>0)\n");
        fprintf(stderr, "    nthreads: number of threads per block (>0)\n");
        fprintf(stderr, "    iters: number of iterations (>0)\n");
}

int main(int argc, char **argv) {
        int blocks, threads, iters;     
        float time;
        cudaEvent_t start, stop;

        if (argc != 4) {
                usage(argv[0]);
                return EXIT_FAILURE;
        }

        blocks = atoi(argv[1]);
        threads = atoi(argv[2]);
        iters = atoi(argv[3]);

        if (blocks <= 0 || threads <= 0 || iters <= 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
        }

        cudaFree(0);
        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        cudaEventRecord(start, 0);
        SyncKernel<<<blocks, threads>>>(iters); 
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        printf("Kernel synchronization overhead time for %d rounds:  %3.5f ms \n", iters, time);
        return EXIT_SUCCESS;
}