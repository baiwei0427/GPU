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

__global__ void blelloch_scan(int *d_out, int *d_in, unsigned int size, bool inclusive)
{
        extern __shared__ int s_data[];
        // thread ID inside the block
        unsigned int tid = threadIdx.x;

        // copy input from global memory to shared memory
        s_data[tid] = d_in[tid];
        __syncthreads();

        // up sweep
        for (int offset = 1; offset < size; offset <<= 1) {
                int tmp = s_data[tid];
                if ((tid + 1) % (offset << 1) == 0) {
                        tmp += s_data[tid - offset];
                }
                __syncthreads();
                s_data[tid] = tmp;
                __syncthreads(); 
        }

        // down sweep
        if (tid == size - 1)
                s_data[tid] = 0;
        __syncthreads();        
        
        for (int offset = size / 2; offset > 0; offset >>= 1) {
                int left, right;
                if ((tid + 1) % (offset << 1) == 0) {
                        left = s_data[tid];
                        right = s_data[tid] + s_data[tid - offset];
                }
                __syncthreads();

                if ((tid + 1) % (offset << 1) == 0) {
                        s_data[tid - offset] = left;
                        s_data[tid] = right;
                }
                __syncthreads();
        }

        // copy output from shared memory to global memory
        // by default, blelloch algorithm generates exclusive scan results 
        if (!inclusive) {
                d_out[tid] = s_data[tid];
        
        // to get inclusive results
        } else {
                if (tid < size - 1)
                        d_out[tid] = s_data[tid + 1];
                else
                        d_out[tid] = s_data[tid] + d_in[tid];
        }
}

int main()
{
        const int iters = 1000;
        const int array_size = 1 << 10;
        int h_in[array_size], h_out[array_size], scan_result[array_size];
        int *d_in, *d_out;

        for (int i = 0; i < array_size; i++) {
                h_in[i] = i + 1;
                // calculate expected inclusive scan result
                if (i == 0) {
                        scan_result[i] = h_in[i];
                } else {
                        scan_result[i] = scan_result[i - 1] + h_in[i];
                }
        }

        // allocate GPU memory
        if (cudaMalloc((void**) &d_in, array_size * sizeof(int)) != cudaSuccess
         || cudaMalloc((void**) &d_out, array_size * sizeof(int)) != cudaSuccess)
                goto out;
        
        // copy the input array from the host memory to the GPU memory
        cudaMemcpy(d_in, h_in, array_size * sizeof(int), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        for (int i = 0; i < iters; i++) {
                //hillissteele_scan<<<1, array_size, array_size * sizeof(int)>>>(d_out, d_in, array_size);
                blelloch_scan<<<1, array_size, array_size * sizeof(int)>>>(d_out, d_in, array_size, true);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);    
        elapsed_time /= iters;      

        printf("Average time elapsed: %f ms\n", elapsed_time);

        // copy the result from the GPU memory to the host memory
        cudaMemcpy(h_out, d_out, array_size * sizeof(int), cudaMemcpyDeviceToHost);


        for (int i = 0; i < array_size; i++) {
                if (h_out[i] != scan_result[i]) {
                        printf("Wrong result\n");
                        goto out;
                }
        }

        printf("Correct result\n");

        /*for (int i = 0; i < array_size; i++) {
                printf("%d ", h_out[i]);
        }

        printf("\n");*/
out:
        cudaFree(d_in);
        cudaFree(d_out);        
        return 0;
}