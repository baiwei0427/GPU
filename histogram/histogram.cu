#include <stdio.h>
#include <sys/time.h>

// do histogram computation on array 'buf' for 'iters' times 
// save results to 'histo'
// return average histogram computation time (per iteration) in millisecond
double cpu_histo(unsigned char *buf, 
                 unsigned int buf_size, 
                 unsigned int *histo, 
                 unsigned int histo_size, 
                 unsigned int iters)
{
        struct timeval start_time, stop_time;

        if (iters == 0 || buf_size == 0 || histo_size == 0)
                return 0;

        gettimeofday(&start_time, NULL);
        // for each round
        for (int i = 0; i < iters; i++) {
                // initialize elements of histo to all 0
                memset(histo, 0, histo_size * sizeof(int));
                // histogram computation
                for (int j = 0; j < buf_size; j++) {
                        histo[buf[j]]++;
                }
        }
        gettimeofday(&stop_time, NULL);

        // measure total time (in millisecond) on GPU
        double total_time_ms = (stop_time.tv_sec - start_time.tv_sec) * 1000 + (stop_time.tv_usec - start_time.tv_usec) / 1000.0;
        // return average per round elapsed time
        return total_time_ms / iters;
}

int main(int argc, char **argv)
{
        // size of histogram
        unsigned int histo_size = 256;
        unsigned int histo[histo_size];
        unsigned int histo_count = 0;

        unsigned int iters = 10;

        // size of array: 100M
        unsigned int array_size = 100 << 20;
        unsigned char *array = (unsigned char*)malloc(array_size);

        if (!array) {
                exit(EXIT_FAILURE);
        }

        srand((unsigned int)time(NULL));
        
        // initialize array with random numbers
        for (int i = 0; i < array_size; i++) {
                // generate a random number in range [0, 255]
                array[i] = rand() & 0xff;
        }

        
        double result = cpu_histo(array, array_size, histo, histo_size, iters);
        printf("Result: %f ms\n", result);

        for (int i = 0; i < histo_size; i++) {
                //printf("%d %u\n", i, histo[i]);
                histo_count += histo[i];
        }

        if (histo_count != array_size) {
                printf("Wrong result\n");
        }

        free(array);
        return 0;
}
