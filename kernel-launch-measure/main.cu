#include <stdio.h>
#include <stdlib.h>

__global__ void EmptyKernel() { }

void usage(char *program)
{       
        fprintf(stderr, "usage: %s nblocks nthreads iters\n", program);
        fprintf(stderr, "    nblocks: number of thread blocks (>0)\n");
        fprintf(stderr, "    nthreads: number of threads per block (>0)\n");
        fprintf(stderr, "    iters: number of iterations (>0)\n");
}

int main(int argc, char **argv) {
        int i, blocks, threads, iters;     
        float time, cumulative_time = 0.f;
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

        for (i = 0; i < iters; i++) { 

                cudaEventRecord(start, 0);
                EmptyKernel<<<blocks, threads>>>(); 
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time, start, stop);
                cumulative_time = cumulative_time + time;
        }

        printf("Kernel launch overhead time:  %3.5f ms \n", cumulative_time / iters);
        return EXIT_SUCCESS;
}