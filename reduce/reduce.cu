#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Interleaved addressing with divergent branching
__global__ void reduce_kernel0(int *d_out, int *d_in)
{
    extern __shared__ int s_data[];

    // thread ID inside the block
    unsigned int tid = threadIdx.x;
    // global ID across all blocks
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy elements from global memoery into per-block shared memory
    s_data[tid] = d_in[gid];
    // Ensure all elements have been copied into shared memory
    __syncthreads();

    // s = 1, 2, 4, 8, ..... blockDim.x / 2
    for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
        if (tid % (s << 1) == 0) {
            s_data[tid] += s_data[tid + s];
        }
        // Ensure all threads in the block finish add in this round
        __syncthreads();
    }

    // write the reduction sum back to the global memory
    if (tid == 0) {
        d_out[blockIdx.x] = s_data[0];
    }
}

// Interleaved addressing with bank conflicts
__global__ void reduce_kernel1(int *d_out, int *d_in)
{
    extern __shared__ int s_data[];

    // thread ID inside the block
    unsigned int tid = threadIdx.x;
    // global ID across all blocks
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy elements from global memoery into per-block shared memory
    s_data[tid] = d_in[gid];
    // Ensure all elements have been copied into shared memory
    __syncthreads();    

    // s = 1, 2, 4, 8, ..... blockDim.x / 2
    for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
        int index = tid * s * 2;

        if (index + s < blockDim.x) {
            s_data[index] += s_data[index + s];
        }

        // Ensure all threads in the block finish add in this round
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = s_data[0];
    }
}

// Sequential addressing
__global__ void reduce_kernel2(int *d_out, int *d_in)
{
    extern __shared__ int s_data[];

    // thread ID inside the block
    unsigned int tid = threadIdx.x;
    // global ID across all blocks
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy elements from global memoery into per-block shared memory
    s_data[tid] = d_in[gid];
    // Ensure all elements have been copied into shared memory
    __syncthreads();    

    // s = blockDim.x / 2, ....., 8, 4, 2, 1
    for (unsigned int s = (blockDim.x >> 1); s >= 1; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        // Ensure all threads in the block finish add in this round
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = s_data[0];
    }
}

// First add during global load
__global__ void reduce_kernel3(int *d_out, int *d_in)
{
    extern __shared__ int s_data[];

    // thread ID inside the block
    unsigned int tid = threadIdx.x;
    // global ID across all blocks
    unsigned int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    s_data[tid] = d_in[gid] + d_in[gid + blockDim.x];
    // Ensure all elements have been copied into shared memory
    __syncthreads();    

    // s = blockDim.x / 2, ....., 8, 4, 2, 1
    for (unsigned int s = (blockDim.x >> 1); s >= 1; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        // Ensure all threads in the block finish add in this round
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = s_data[0];
    }    
}

__device__ void warpReduce4(volatile int* s_data, int tid) 
{
    s_data[tid] += s_data[tid + 32];
    s_data[tid] += s_data[tid + 16];
    s_data[tid] += s_data[tid + 8];
    s_data[tid] += s_data[tid + 4];
    s_data[tid] += s_data[tid + 2];
    s_data[tid] += s_data[tid + 1];
}

// Unroll the last warp
__global__ void reduce_kernel4(int *d_out, int *d_in)
{
    extern __shared__ int s_data[];

    // thread ID inside the block
    unsigned int tid = threadIdx.x;
    // global ID across all blocks
    unsigned int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    s_data[tid] = d_in[gid] + d_in[gid + blockDim.x];
    // Ensure all elements have been copied into shared memory
    __syncthreads();    

    // s = blockDim.x / 2, ....., 128, 64
    for (unsigned int s = (blockDim.x >> 1); s > 32; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        // Ensure all threads in the block finish add in this round
        __syncthreads();
    }

    // now we only have 32 active threads (a warp) left
    if (tid < 32) {
        warpReduce4(s_data, tid);
    }

    if (tid == 0) {
        d_out[blockIdx.x] = s_data[0];
    }    
}

template <unsigned int block_size> __device__ void warpReduce5(volatile int* s_data, int tid) 
{
    if (block_size >= 64) {
        s_data[tid] += s_data[tid + 32];
    }

    if (block_size >= 32) {
        s_data[tid] += s_data[tid + 16];
    }

    if (block_size >= 16) {
        s_data[tid] += s_data[tid + 8];
    }

    if (block_size >= 8) {
        s_data[tid] += s_data[tid + 4];
    }

    if (block_size >= 4) {
        s_data[tid] += s_data[tid + 2];
    }

    if (block_size >= 2) {
        s_data[tid] += s_data[tid + 1];
    }
}

// Completely unrolled
// The block size is limited to 1024 threads.
// Also, we are sticking to power-of-2 block sizesSo
template <unsigned int block_size> __global__ void reduce_kernel5(int *d_out, int *d_in)
{
    extern __shared__ int s_data[];

    // thread ID inside the block
    unsigned int tid = threadIdx.x;
    // global ID across all blocks
    unsigned int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    s_data[tid] = d_in[gid] + d_in[gid + blockDim.x];
    // Ensure all elements have been copied into shared memory
    __syncthreads();

    if (block_size >= 1024) {
        if (tid < 512) { 
            s_data[tid] += s_data[tid + 512]; 
        }
        __syncthreads(); 
    } 
    
    if (block_size >= 512) {
        if (tid < 256) { 
            s_data[tid] += s_data[tid + 256]; 
        }
        __syncthreads(); 
    } 

    if (block_size >= 256) {
        if (tid < 128) { 
            s_data[tid] += s_data[tid + 128]; 
        }
        __syncthreads(); 
    } 

    if (block_size >= 128) {
        if (tid < 64) { 
            s_data[tid] += s_data[tid + 64]; 
        }
        __syncthreads(); 
    }

    // only a warp left
    if (tid < 32) {
        warpReduce5<block_size>(s_data, tid); 
    }

    if (tid == 0) {
        d_out[blockIdx.x] = s_data[0];
    }   
}

// wrapper function to run reduce_kernel5
void run_reduce_kernel5(int *d_out, int *d_in, unsigned int blocks, unsigned int threads)
{
    //printf("threads = %u\n", threads);
    switch (threads) {
        case 1024:
            reduce_kernel5<1024><<<blocks, threads, threads * sizeof(int)>>>(d_out, d_in);
            break;
        case 512:
            reduce_kernel5< 512><<<blocks, threads, threads * sizeof(int)>>>(d_out, d_in);
            break;
        case 256:
            reduce_kernel5< 256><<<blocks, threads, threads * sizeof(int)>>>(d_out, d_in);        
            break;
        case 128:
            reduce_kernel5< 128><<<blocks, threads, threads * sizeof(int)>>>(d_out, d_in);        
            break;
        case 64:
            reduce_kernel5<  64><<<blocks, threads, threads * sizeof(int)>>>(d_out, d_in);        
            break;
        case 32:
            reduce_kernel5<  32><<<blocks, threads, threads * sizeof(int)>>>(d_out, d_in);
            break;
        case 16:
            reduce_kernel5<  16><<<blocks, threads, threads * sizeof(int)>>>(d_out, d_in);        
            break;
        case 8:
            reduce_kernel5<   8><<<blocks, threads, threads * sizeof(int)>>>(d_out, d_in);        
            break;
        case 4:
            reduce_kernel5<   4><<<blocks, threads, threads * sizeof(int)>>>(d_out, d_in);        
            break;
        case 2:
            reduce_kernel5<   2><<<blocks, threads, threads * sizeof(int)>>>(d_out, d_in);        
            break;
        case 1:
            reduce_kernel5<   1><<<blocks, threads, threads * sizeof(int)>>>(d_out, d_in);        
            break;
        default:
            printf("Invalid block size %u\n", threads);                        
    }
}

inline bool is_power_of_2(int n)
{
    return ((n & (n - 1)) == 0);
}

void print_kernel_info(int kernel_id) 
{
    switch (kernel_id) {
        case 0: 
            printf("Interleaved addressing with divergent branching\n");
            break;
        case 1:
            printf("Interleaved addressing with bank conflicts\n");
            break;  
        case 2:
            printf("Sequential addressing\n");
            break;
        case 3:
            printf("First add during global load\n");
            break;
        case 4:
            printf("Unroll last warp\n");
            break;
        case 5:
            printf("Completely unrolled\n");
            break;              
        default:
            printf("Invalid kernel function ID %d\n", kernel_id);        
    }    
}

// input: array (in host memory), array size, expected result, kernel function ID and iterations 
void reduce(int *h_in, int array_size, int expected_result, int kernel_id, int iters)
{
    // # of threads per block. It should be the power of two
    int threads = 1 << 10;
    // # of blocks in total. 
    int blocks = 1;
    // GPU memory pointers
    int *d_in, *d_intermediate, *d_out;
    // final result in host memory
    int h_out;
    // events to record start and stop time
    cudaEvent_t start, stop;
    // elapsed time
    float elapsed_time;

    // print kernel information
    print_kernel_info(kernel_id); 
    
    if (!h_in || array_size <= 0 || !is_power_of_2(array_size))
        goto out;

    if (array_size > threads)
        blocks = array_size / threads;
    
    // allocate GPU memory
    if (cudaMalloc((void**) &d_in, array_size * sizeof(int)) != cudaSuccess
     || cudaMalloc((void**) &d_intermediate, blocks * sizeof(int)) != cudaSuccess
     || cudaMalloc((void**) &d_out, sizeof(int)) != cudaSuccess)
        goto out;

    // copy the input array from the host memory to the GPU memory
    cudaMemcpy(d_in, h_in, array_size * sizeof(int), cudaMemcpyHostToDevice);

    // create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // run many times
    for (int i = 0; i < iters; i++) {
        switch (kernel_id) {
            // Interleaved addressing with divergent branching 
            case 0: 
                reduce_kernel0<<<blocks, threads, threads * sizeof(int)>>>(d_intermediate, d_in); 
                reduce_kernel0<<<1, blocks, blocks * sizeof(int)>>>(d_out, d_intermediate);
                break;
            // Interleaved addressing with bank conflicts
            case 1:
                reduce_kernel1<<<blocks, threads, threads * sizeof(int)>>>(d_intermediate, d_in);
                reduce_kernel1<<<1, blocks, blocks * sizeof(int)>>>(d_out, d_intermediate);
                break;  
            // Sequential addressing              
            case 2:
                reduce_kernel2<<<blocks, threads, threads * sizeof(int)>>>(d_intermediate, d_in);
                reduce_kernel2<<<1, blocks, blocks * sizeof(int)>>>(d_out, d_intermediate);
                break;
            // First add during global load
            case 3:
                reduce_kernel3<<<blocks, threads / 2 , threads / 2 * sizeof(int)>>>(d_intermediate, d_in);
                reduce_kernel3<<<1, blocks / 2, blocks / 2 * sizeof(int)>>>(d_out, d_intermediate);  
                break;
            // Unroll last warp
            case 4:
                reduce_kernel4<<<blocks, threads / 2 , threads / 2 * sizeof(int)>>>(d_intermediate, d_in);
                reduce_kernel4<<<1, blocks / 2, blocks / 2 * sizeof(int)>>>(d_out, d_intermediate);
                break;
            // Completely unrolled
            case 5:
                run_reduce_kernel5(d_intermediate, d_in, blocks, threads / 2);
                run_reduce_kernel5(d_out, d_intermediate, 1, blocks / 2);               
                break;                          
            default:
                printf("Invalid kernel function ID %d\n", kernel_id);   
                goto out;      
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);    
    elapsed_time /= iters;      
    printf("Average time elapsed: %f ms\n", elapsed_time);

    // destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // copy the result from the GPU memory to the host memory
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_out != expected_result) {
        printf("Wrong result: %d (expected) %d (actual)\n", expected_result, h_out);
    }

out:
    // free GPU memory
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);
}

// generate a random integer in [min, max]
inline int random_range(int min, int max)
{
    if (min > max)
        return 0;
    else
        return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

int main(int argc, char **argv) 
{
    if (argc != 3) {
        printf("%s [kernel ID] [iterations]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int kernel_id = atoi(argv[1]);
    int iters = atoi(argv[2]);
    if (iters <= 0 || kernel_id < 0) {
        printf("Invalid input\n");
        exit(EXIT_FAILURE);
    }

    const int ARRAY_SIZE = 1 << 20;
    int h_in[ARRAY_SIZE];
    int sum = 0;
    
    // initialize random number generator
    srand(time(NULL));
    int min = 0, max = 10;

    for (int i = 0; i < ARRAY_SIZE; i++) {
        // generate a random int in a range
        h_in[i] = random_range(min, max);
        sum += h_in[i];
    }

    // launch reduce kernels
    reduce(h_in, ARRAY_SIZE, sum, kernel_id, iters);

    return 0;
}