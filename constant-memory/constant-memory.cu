#include <stdio.h>

#define ANGLE_COUNT 360

// declare constant memory
__constant__ float cangle[ANGLE_COUNT];
// declare global memory
__device__ float gangle[ANGLE_COUNT];

// kernel function for constant memory
__global__ void test_kernel(float* darray)
{
        int index = blockIdx.x * blockDim.x + threadIdx.x;
   
        for (int loop = 0; loop < ANGLE_COUNT; loop++) {
                darray[index] = darray[index] + cangle[loop] ;
        }
}

// kernel function for global memory
__global__ void test_kernel2(float* darray)
{
        int index = blockIdx.x * blockDim.x + threadIdx.x;
   
        for (int loop = 0; loop < ANGLE_COUNT; loop++) {
                darray[index] = darray[index] + gangle[loop] ;
        }
}

int main(int argc,char** argv)
{
        int threads_per_block = 256;
        int blocks = 32;
        int size = blocks * threads_per_block;
        float* darray;
        float hangle[360];
        cudaEvent_t startEvent, stopEvent;
        float time;

        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        //initialize angle array on host
        for (int loop = 0; loop < ANGLE_COUNT; loop++) {
               hangle[loop] = acos( -1.0f )* loop / 180.0f;
        }

        //allocate device memory
        cudaMalloc((void**)&darray, sizeof(float) * size);
         
        //initialize allocated memory
        cudaMemset(darray, 0, sizeof(float) * size);

        //copy host angle data to constant memory
        cudaMemcpyToSymbol(cangle, hangle, sizeof(float) * ANGLE_COUNT);
        
        cudaEventRecord(startEvent, 0);
        test_kernel<<<blocks, threads_per_block>>>(darray);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        printf("Time for constant memory (ms): %f\n", time);

        //re-initialize allocated memory
        cudaMemset(darray, 0, sizeof(float) * size);

        //copy host angle data to global memory
        cudaMemcpy(gangle, hangle, sizeof(float) * ANGLE_COUNT, cudaMemcpyHostToDevice);
        
        cudaEventRecord(startEvent, 0);
        test_kernel2<<<blocks, threads_per_block>>>(darray);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        printf("Time for global memory (ms): %f\n", time);

        //free device memory
        cudaFree(darray);

        //destroy eventsmake
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);

        return 0;
}
