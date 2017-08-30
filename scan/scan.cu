#include <stdio.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__global__ void hillissteele_scan_kernel(int *d_out, int *d_in, int *block_sum)
{
        extern __shared__ int s_data[];

        // ID inside the block
        int tid = threadIdx.x;
        // global ID across blocks
        int gid = threadIdx.x + blockIdx.x * blockDim.x;

        // copy input from global memory to shared memory
        s_data[tid] = d_in[gid];
        __syncthreads();

        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
                int tmp = s_data[tid];
                if (tid >= offset) {
                        tmp += s_data[tid - offset];
                }
                __syncthreads();
                s_data[tid] = tmp;
                __syncthreads(); 
        }

        // copy output from shared memory to global memory
        d_out[gid] = s_data[tid];

        // if this is the last thread in the block, write the sum of the block
        if (block_sum && tid == blockDim.x - 1) { 
                block_sum[blockIdx.x] = s_data[tid];
                //printf("Sum of block %d: %d\n", blockIdx.x, block_sum[blockIdx.x]);
        }
}

__global__ void hillissteele_scan2_kernel(int *d_out, int *d_in, int *block_sum)
{
        extern __shared__ int s_data[];

        // ID inside the block
        int tid = threadIdx.x;
        // global ID across blocks
        int gid = threadIdx.x + blockIdx.x * blockDim.x;

        int in = 0, out = 1;

        // copy input from global memory to shared memory
        // s_data actually has two arrays: in and out
        s_data[out * blockDim.x + tid] = d_in[gid];
        __syncthreads();

       for (int offset = 1; offset < blockDim.x; offset <<= 1) {
                // swap in and out
                in = out;
                out = 1 - in;

                s_data[out * blockDim.x + tid] = s_data[in * blockDim.x + tid];
                if (tid >= offset) {
                        s_data[out * blockDim.x + tid] += s_data[in * blockDim.x + tid - offset];
                } 

                __syncthreads();
        }

        // copy output from shared memory to global memory
        d_out[gid] = s_data[out * blockDim.x + tid];

        // if this is the last thread in the block, write the sum of the block
        if (block_sum && tid == blockDim.x - 1) { 
                block_sum[blockIdx.x] = s_data[out * blockDim.x + tid];
                //printf("Sum of block %d: %d\n", blockIdx.x, block_sum[blockIdx.x]);
        }
}

__global__ void blelloch_scan_kernel(int *d_out, int *d_in, int *block_sum)
{
        extern __shared__ int s_data[];
        // ID inside the block
        int tid = threadIdx.x;
        // global ID across blocks
        int gid = threadIdx.x + blockIdx.x * blockDim.x;

        // copy input into shared memory. each thread copies 2 elements.
        s_data[2 * tid] = d_in[2 * gid];
        s_data[2 * tid + 1] = d_in[2 * gid + 1];

        // up-sweep
        int offset = 1;
        for (int d = blockDim.x; d > 0; d >>= 1) {
                if (tid < d) {
                        int index = 2 * offset * (tid + 1) - 1;
                        s_data[index] += s_data[index - offset];
                }
                __syncthreads();
                offset <<= 1;
        }

        // clear the element
        if (tid == 0) {
                s_data[2 * blockDim.x - 1] = 0; 
        }  

        offset = blockDim.x;
        // down-sweep
        for (int d = 1; d <= blockDim.x; d <<= 1) {
                if (tid < d) {
                        int index = 2 * offset * (tid + 1) - 1;
                        int tmp = s_data[index];
                        s_data[index] += s_data[index - offset];
                        s_data[index - offset] = tmp;
                }
                __syncthreads();
                offset >>= 1;
        }
               
        // write results to device memory
        d_out[2 * gid] = s_data[2 * tid + 1];

        // if it is the last thread in the block
        if (tid == blockDim.x - 1) {
                d_out[2 * gid + 1] = s_data[2 * tid + 1] + d_in[2 * gid + 1];
                // save the block sum
                if (block_sum) {
                      block_sum[blockIdx.x] = d_out[2 * gid + 1]; 
                      //printf("Sum of block %d: %d\n", blockIdx.x, block_sum[blockIdx.x]); 
                }

        } else {
                d_out[2 * gid + 1] = s_data[2 * tid + 2];
        }
}

__global__ void add_kernel(int *d_out, int *block_sum, int elements_per_thread)
{
        int bid = blockIdx.x;
        int gid = threadIdx.x + blockIdx.x * blockDim.x;
        int tmp; 
        
        if (bid == 0)
                return;

        tmp = block_sum[bid - 1];

        // add scanned block sum i to all values of scanned block i + 1       
        for (int i = 0; i < elements_per_thread; i++) {
                d_out[elements_per_thread * gid + i] += tmp;
        }
}

void scan(int *h_out, int *h_in, int array_size, int max_threads_per_block, int kernel_id, int iters, int *expected_result)
{
        int *d_in, *d_out, *block_sum;

        // # of threads. blelloch scan only needs array_size / 2 threads in total
        int threads = (kernel_id == 2)? array_size / 2: array_size;
        // # of blocks
        int blocks = MAX(threads / max_threads_per_block, 1);
        // # of threads per block
        int threads_per_block = MIN(max_threads_per_block, threads);
        
        printf("Blocks %d  Threads per block %d\n", blocks, threads_per_block);

        // shared memory in bytes
        int shem = threads_per_block * sizeof(int);
        if (kernel_id > 0) {
                shem *= 2;
        }

        // allocate GPU memory
        if (cudaMalloc((void**) &d_in, array_size * sizeof(int)) != cudaSuccess
         || cudaMalloc((void**) &d_out, array_size * sizeof(int)) != cudaSuccess
         || cudaMalloc((void**) &block_sum, blocks * sizeof(int)) != cudaSuccess)
                goto out;
        
        // copy the input array from the host memory to the GPU memory
        cudaMemcpy(d_in, h_in, array_size * sizeof(int), cudaMemcpyHostToDevice);        

        for (int i = 0; i < iters; i++) {
                // scan each block                                
                switch (kernel_id) {
                        case 0:
                                hillissteele_scan_kernel<<<blocks, threads_per_block, shem>>>(d_out, d_in, block_sum);
                                break;
                        case 1:
                                hillissteele_scan2_kernel<<<blocks, threads_per_block, shem>>>(d_out, d_in, block_sum);
                                break;
                        case 2:
                                blelloch_scan_kernel<<<blocks, threads_per_block, shem>>>(d_out, d_in, block_sum);
                                break;
                        default:
                                printf("Invalid kernel function ID %d\n", kernel_id);
                                goto out;
                }
                
                // if there is only a single block, skip all rest steps
                if (blocks == 1) {
                        continue;
                }

                // scan block sums (blocks elements in total)
                switch (kernel_id) {
                        case 0:
                                hillissteele_scan_kernel<<<1, blocks, blocks * sizeof(int)>>>(block_sum, block_sum, NULL);
                                break;
                        case 1:
                                hillissteele_scan2_kernel<<<1, blocks, 2 * blocks * sizeof(int)>>>(block_sum, block_sum, NULL);
                                break;
                        case 2:
                                blelloch_scan_kernel<<<1, blocks / 2, blocks * sizeof(int)>>>(block_sum, block_sum, NULL);
                                break;
                        default:
                                break;
                }

                int elements_per_thread = 1;
                if (kernel_id == 2)
                        elements_per_thread = 2;

                // add scanned block sum i to all values of scanned block i + 1
                add_kernel<<<blocks, threads_per_block, 0>>>(d_out, block_sum, elements_per_thread);
        }

        // copy the result from the GPU memory to the host memory
        cudaMemcpy(h_out, d_out, array_size * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < array_size; i++) {
                //printf("%d ", h_out[i]);
                if (h_out[i] != expected_result[i]) {
                        printf("Wrong result\n");
                        goto out;
                }
        }

        printf("Correct result\n");   

out:
        /*for (int i = 0; i < array_size; i++) {
                printf("%d ", h_out[i]);
        }

        printf("\n");*/

        cudaFree(d_in);
        cudaFree(d_out);     
        cudaFree(block_sum);
}

// generate a random integer in [a, b]
inline int random_range(int a, int b)
{
    if (a > b)
        return 0;
    else
        return a + rand() / (RAND_MAX / (b - a + 1) + 1);
}

int main(int argc, char **argv)
{
        const int iters = 10000;
        const int array_size = 1 << 16;
        const int threads_per_block = 1 << 10;
        int *h_in, *h_out, *scan_result;
        int kernel_id = 0;

        if (argc != 2) {
                printf("Usage: %s [kernel ID]\n", argv[0]);
                exit(1);
        } else {
                kernel_id = atoi(argv[1]);
        }

        h_in = (int*)malloc(array_size * sizeof(int));
        h_out = (int*)malloc(array_size * sizeof(int));
        scan_result = (int*)malloc(array_size * sizeof(int));

        // no enough memory
        if (!h_in || !h_out || !scan_result)
                goto out;

        // initialize random number generator
        srand(time(NULL));

        for (int i = 0; i < array_size; i++) {
                //h_in[i] = 1;
                h_in[i] = random_range(0, 10);
                // calculate expected inclusive scan result
                if (i == 0) {
                        scan_result[i] = h_in[i];
                } else {
                        scan_result[i] = scan_result[i - 1] + h_in[i];
                }
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        scan(h_out, h_in, array_size, threads_per_block, kernel_id, iters, scan_result);
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);    
        elapsed_time /= iters;      

        printf("Average time elapsed: %f ms\n", elapsed_time);

out:
        free(h_in);
        free(h_out);
        free(scan_result);

        return 0;
}