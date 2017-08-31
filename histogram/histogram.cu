#include <stdio.h>
#include <sys/time.h>

// do histogram computation on array 'buf' with 'size' bytes, save results to 'histo'
void cpu_histo(unsigned char *buf, unsigned int size, unsigned int *histo)
{
        for (int i = 0; i < size; i++) {
                histo[buf[i]]++;
        }
}

int main(int argc, char **argv)
{
        // size of array: 100M
        unsigned int array_size = 100 << 20;
        // size of histogram
        unsigned int histo_size = 256;
        unsigned char *buffer = (unsigned char*)malloc(array_size);
        unsigned int histo[histo_size];
        unsigned int histo_count = 0;
        struct timeval start_time, stop_time;

        if (!buffer) {
                exit(EXIT_FAILURE);
        }

        srand((unsigned int)time(NULL));
        
        // initialize array with random numbers
        for (int i = 0; i < array_size; i++) {
                // generate a random number in range [0, 255]
                buffer[i] = rand() & 0xff;
        }

        // initialize histogram results 
        for (int i = 0; i < histo_size; i++) {
                histo[i] = 0;
        }

        gettimeofday(&start_time, NULL);
        
        cpu_histo(buffer, array_size, histo);
        
        gettimeofday(&stop_time, NULL);

        // measure histogram computation time (in millisecond) on GPU
        double elapsed_time_ms = (stop_time.tv_sec - start_time.tv_sec) * 1000 + (stop_time.tv_usec - start_time.tv_usec)/1000.0;
        printf("Time elapsed: %.2f ms\n", elapsed_time_ms);

        for (int i = 0; i < histo_size; i++) {
                //printf("%d %u\n", i, histo[i]);
                histo_count += histo[i];
        }

        if (histo_count != array_size) {
                printf("Wrong result\n");
        }

        free(buffer);
        return 0;
}