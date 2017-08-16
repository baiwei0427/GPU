#include <stdio.h>

void print_device_info(cudaDeviceProp prop)
{
        printf("Name: %s\n", prop.name);
        printf("Clock (KHz): %d\n", prop.memoryClockRate);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
        printf("Maximum size of each dimension of a grid: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Maximum size of each dimension of a block: %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Shared memory available per block in bytes: %d\n", prop.sharedMemPerBlock);
}

int main()
{
        int device_count;
        cudaGetDeviceCount(&device_count);
        printf("Total number of devices: %d\n", device_count);
        
        for (int i = 0; i < device_count; i++) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                printf("==================== Device %d ====================\n", i);
                print_device_info(prop);
        }

        return 0;
}