#include <stdio.h>
#include <stdlib.h>

void usage(char *program)
{     
        fprintf(stderr, "usage: %s memsize iters\n", program);
        fprintf(stderr, "    memsize: memory allocated in bytes (>0)\n");
        fprintf(stderr, "    iters: number of iterations (>0)\n");
}

int main(int argc, char **argv)
{
        int size, iters, i;
        void *ptr;
        float time, cumulative_malloc_time = 0.f, cumulative_free_time = 0.f;
        cudaEvent_t start, stop;

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

        cudaFree(0);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (i = 0; i < iters; i++) {
                
                cudaEventRecord(start, 0);
                if (cudaMalloc(&ptr, size) != cudaSuccess) {
                        fprintf(stderr, "cudaMalloc error\n");
                }
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time, start, stop);
                cumulative_malloc_time += time;                                                

                cudaEventRecord(start, 0);                
                if (cudaFree(ptr) != cudaSuccess) {
                        fprintf(stderr, "cudaFree error\n");                        
                }
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time, start, stop);
                cumulative_free_time += time;
        }

        printf("cudaMalloc time:  %3.5f ms \n", cumulative_malloc_time / iters);
        printf("cudaFree   time:  %3.5f ms \n", cumulative_free_time / iters);

        return EXIT_SUCCESS;
}