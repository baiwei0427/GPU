#include <stdio.h>

void print_device_info(cudaDeviceProp prop)
{
        printf("Name: %s\n", prop.name);

        printf("Clock (KHz): %d\n", prop.clockRate);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
        
        printf("Maximum size of each dimension of a grid: %d %d %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Maximum size of each dimension of a block: %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
        
        printf("Global memory available on device in bytes: %u\n", prop.totalGlobalMem);
        printf("Shared memory available per block in bytes: %d\n", prop.sharedMemPerBlock);
        printf("Shared memory available per multiprocessor in bytes: %d\n", prop.sharedMemPerMultiprocessor);
        printf("Size of L2 cache in bytes: %d\n", prop.l2CacheSize);

        printf("Peak memory clock frequency (KHz): %d\n", prop.memoryClockRate);
        printf("Global memory bus width in bits: %d\n", prop.memoryBusWidth);
        printf("Peak memory bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
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