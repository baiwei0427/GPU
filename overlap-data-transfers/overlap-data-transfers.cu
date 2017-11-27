#include <stdio.h>
#include <stdlib.h>
#include "helper_cuda.h"

// print the usage of the program
void usage(char *program);
// kernel function
__global__ void kernel(float *a, int offset);

int main(int argc, char **argv) 
{
        int devID, nStreams;
        cudaDeviceProp prop;
        float *h_array, *d_array;
        int threadsPerBlock = 256;      // # of threads for each block
        int blocksPerStream = 1024;     // # of blocks for each stream
        int n, bytes, streamSize, streamBytes;  

        if (argc != 3) {
                usage(argv[0]);
                return EXIT_FAILURE;
        }

        devID = atoi(argv[1]);          // device ID
        nStreams = atoi(argv[2]);       // # of streams

        if (devID < 0 || nStreams <= 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
        }

        n = nStreams * blocksPerStream * threadsPerBlock;       // total # of threads / elements
        bytes = n * sizeof(float);                              // total # of bytes
        streamSize = n / nStreams;                              // # of threads / elements for each stream
        streamBytes = bytes / nStreams;                         // # of bytes for each stream

        // Print device information
        checkCudaErrors(cudaGetDeviceProperties(&prop, devID));
        printf("Device : %s\n", prop.name);
        printf("Concurrent copy and execution: %s with %d copy engines\n",  
                (prop.deviceOverlap ? "Yes" : "No"), 
                prop.asyncEngineCount);

        // Choose GPU device
        checkCudaErrors(cudaSetDevice(devID));

        // Allocate pinned host memory and device memory
        checkCudaErrors(cudaMallocHost((void**)&h_array, bytes));      
        checkCudaErrors(cudaMalloc((void**)&d_array, bytes));

        // Create events and streams
        cudaEvent_t startEvent, stopEvent;
        cudaStream_t stream[nStreams];
        checkCudaErrors(cudaEventCreate(&startEvent));
        checkCudaErrors(cudaEventCreate(&stopEvent));
        for (int i = 0; i < nStreams; ++i)
                checkCudaErrors(cudaStreamCreate(&stream[i]));

        float time;     // elapsed time

        // Baseline case - sequential transfer and execute
        checkCudaErrors(cudaEventRecord(startEvent,0));
        checkCudaErrors(cudaMemcpy(d_array, h_array, bytes, cudaMemcpyHostToDevice));
        kernel<<<n / threadsPerBlock, threadsPerBlock>>>(d_array, 0);
        checkCudaErrors(cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaEventRecord(stopEvent, 0));

        checkCudaErrors(cudaEventSynchronize(stopEvent)); // wait for stopEvent
        checkCudaErrors(cudaEventElapsedTime(&time, startEvent, stopEvent));
        printf("Time for sequential transfer and execute (ms): %f\n", time);

        // Asynchronous version 1: loop over {copy, kernel, copy}
        checkCudaErrors(cudaEventRecord(startEvent,0));
        for (int i = 0; i < nStreams; i++) {
                int offset = i * streamSize;
                checkCudaErrors(cudaMemcpyAsync(&d_array[offset], 
                                                &h_array[offset], 
                                                streamBytes, 
                                                cudaMemcpyHostToDevice, 
                                                stream[i]));
                kernel<<<streamSize / threadsPerBlock, threadsPerBlock, 0, stream[i]>>>(d_array, offset);
                checkCudaErrors(cudaMemcpyAsync(&h_array[offset], 
                                                &d_array[offset], 
                                                streamBytes, 
                                                cudaMemcpyDeviceToHost, 
                                                stream[i]));
        }

        checkCudaErrors(cudaEventRecord(stopEvent, 0));
        checkCudaErrors(cudaEventSynchronize(stopEvent)); // wait for stopEvent
        checkCudaErrors(cudaEventElapsedTime(&time, startEvent, stopEvent));
        printf("Time for asynchronous version 1 (ms): %f\n", time);

        // Asynchronous version 2: loop over copy, loop over kenel, loop over copy
        checkCudaErrors(cudaEventRecord(startEvent,0));
        for (int i = 0; i < nStreams; i++) {
                int offset = i * streamSize;
                checkCudaErrors(cudaMemcpyAsync(&d_array[offset], 
                                                &h_array[offset], 
                                                streamBytes, 
                                                cudaMemcpyHostToDevice, 
                                                stream[i]));
        }

        for (int i = 0; i < nStreams; i++) {
                int offset = i * streamSize;
                kernel<<<streamSize / threadsPerBlock, threadsPerBlock, 0, stream[i]>>>(d_array, offset);
        }

        for (int i = 0; i < nStreams; i++) {
                int offset = i * streamSize;
                checkCudaErrors(cudaMemcpyAsync(&h_array[offset], 
                                                &d_array[offset], 
                                                streamBytes, 
                                                cudaMemcpyDeviceToHost, 
                                                stream[i]));
        }

        checkCudaErrors(cudaEventRecord(stopEvent, 0));
        checkCudaErrors(cudaEventSynchronize(stopEvent)); // wait for stopEvent
        checkCudaErrors(cudaEventElapsedTime(&time, startEvent, stopEvent));
        printf("Time for asynchronous version 2 (ms): %f\n", time);

        // Destroy events and streams
        checkCudaErrors(cudaEventDestroy(startEvent));
        checkCudaErrors(cudaEventDestroy(stopEvent));
        for (int i = 0; i < nStreams; ++i)
                checkCudaErrors(cudaStreamDestroy(stream[i]));

        // Free memory
        cudaFreeHost(h_array);
        cudaFree(d_array);

        return EXIT_SUCCESS;
}

void usage(char *program) 
{
        printf("%s devID nStreams\n", program);
}

__global__ void kernel(float *a, int offset)
{
        int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
        float x = (float)i;
        float s = sinf(x); 
        float c = cosf(x);
        a[i] = sqrtf(s * s + c * c);
}