#include <stdio.h>
#include <helper_cuda.h>

__global__ void saxpy(int n, float a, float *x, float *y)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) 
                y[i] = a * x[i] + y[i];
}

int main(void)
{
        int N = 1 << 20;
        float *x, *y, *d_x, *d_y;
        
        if (!(x = (float*)malloc(N * sizeof(float)))) {
                return 0;
        }

        if (!(y = (float*)malloc(N * sizeof(float)))) {
                return 0;
        }

        checkCudaErrors(cudaMalloc(&d_x, N * sizeof(float))); 
        checkCudaErrors(cudaMalloc(&d_y, N * sizeof(float)));

        for (int i = 0; i < N; i++) {
                x[i] = 1.0f;
                y[i] = 2.0f;
        }     

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        checkCudaErrors(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));

        cudaEventRecord(start);
        // Perform SAXPY on 1M elements
        saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);
        cudaEventRecord(stop);

        checkCudaErrors(cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
        
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        float maxError = 0.0f;
        for (int i = 0; i < N; i++)
                maxError = max(maxError, abs(y[i] - 4.0f));
        
        printf("Max error: %f\n", maxError);
        printf("Elapsed time: %f ms\n", milliseconds);

        cudaFree(d_x);
        cudaFree(d_y);
        free(x);
        free(y);
}