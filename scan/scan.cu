#include <stdio.h>

__global__ void hillissteele_scan(int *d_out, int *d_in, unsigned int size)
{
        extern __shared__ int s_data[];
        // thread ID inside the block
        unsigned int tid = threadIdx.x;

        // copy input from global memory to shared memory
        s_data[tid] = d_in[tid];
        __syncthreads();

        for (int offset = 1; offset < size; offset <<= 1) {
                int tmp = s_data[tid];
                if (tid >= offset) {
                        tmp += s_data[tid - offset];
                }
                __syncthreads();
                s_data[tid] = tmp;
                __syncthreads(); 
        }

        // copy output from shared memory to global memory
        d_out[tid] = s_data[tid];
}

int main()
{
        const int array_size = 1 << 10;
        int h_in[array_size], h_out[array_size];
        int *d_in, *d_out;

        for (int i = 0; i < array_size; i++) {
                h_in[i] = i;
        }

        // allocate GPU memory
        if (cudaMalloc((void**) &d_in, array_size * sizeof(int)) != cudaSuccess
         || cudaMalloc((void**) &d_out, array_size * sizeof(int)) != cudaSuccess)
                goto out;
        
        // copy the input array from the host memory to the GPU memory
        cudaMemcpy(d_in, h_in, array_size * sizeof(int), cudaMemcpyHostToDevice);

        hillissteele_scan<<<1, array_size, array_size * sizeof(int)>>>(d_out, d_in, array_size);

        // copy the result from the GPU memory to the host memory
        cudaMemcpy(h_out, d_out, array_size * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < array_size; i++) {
                printf("%d ", h_out[i]);
        }
        printf("\n");
out:
        cudaFree(d_in);
        cudaFree(d_out);        
        return 0;
}