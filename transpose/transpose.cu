#include <stdio.h>
#include <sys/time.h>
#include "helper_cuda.h"

const int N = 1024;     // matrix size is N x N
const int K = 32;       // tile size is K x K 

void transpose_cpu(int *in, int *out) 
{
        for (int row = 0; row < N; row++) {
                for (int col = 0; col < N; col++) {
                        out[col * N + row] = in[row * N + col];
                }
        }
}

__global__ void transpose_serial(int *in, int *out) 
{
        for (int row = 0; row < N; row++) {
                for (int col = 0; col < N; col++) {
                        out[col * N + row] = in[row * N + col];
                }
        }
}

__global__ void transpose_parallel_per_row(int *in, int *out)
{
        int row = threadIdx.x;

        for (int col = 0; col < N; col++) {
                out[col * N + row] = in[row * N + col];
        }
}

__global__ void transpose_parallel_per_element(int *in, int *out)
{
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        out[col * N + row] = in[row * N + col];
}

__global__ void transpose_parallel_per_element_tiled(int *in, int *out)
{       
        __shared__ int s_data[K][K];
        int x = threadIdx.x, y = threadIdx.y;

        int in_corner_x = blockIdx.x * K, in_corner_y = blockIdx.y * K;
        int out_corner_x = in_corner_y, out_corner_y = in_corner_x; 

        // write in[y][x] to s_data[y][x]
        s_data[y][x] = in[(in_corner_y + y) * N + (in_corner_x + x)];
        __syncthreads();

        // write s_data[x][y] to out[y][x] 
        out[(out_corner_y + y) * N + (out_corner_x + x)] = s_data[x][y];
}

__global__ void transpose_parallel_per_element_tiled16(int *in, int *out)
{       
        __shared__ int s_data[16][16];
        int x = threadIdx.x, y = threadIdx.y;

        int in_corner_x = blockIdx.x * 16, in_corner_y = blockIdx.y * 16;
        int out_corner_x = in_corner_y, out_corner_y = in_corner_x; 

        // write in[y][x] to s_data[y][x]
        s_data[y][x] = in[(in_corner_y + y) * N + (in_corner_x + x)];
        __syncthreads();

        // write s_data[x][y] to out[y][x] 
        out[(out_corner_y + y) * N + (out_corner_x + x)] = s_data[x][y];
}

void print_matrix(int *in) 
{
        for (int row = 0; row < N; row++) {
                for (int col = 0; col < N; col++) {
                        printf("%d ", in[row * N + col]);
                }
                printf("\n");
        }
}

void fill_matrix(int *in) 
{
        int size = N * N;
        for (int i = 0; i < size; i++) {
                in[i] = rand() % 10;
        }
}

// return (matrix a == matrix b)
bool same_matrices(int *a, int *b)
{
        int size = N * N;
        for (int i = 0; i < size; i++) {
                if (a[i] != b[i]) {
                        return false;
                }
        }
        return true;
}

int main(int argc, char **argv) 
{
        cudaEvent_t start, stop;
        struct timeval start_time, stop_time;
        float elapsed_time;
        int num_bytes = N * N * sizeof(int);
        int *h_in = (int*)malloc(num_bytes);
        int *h_out = (int*)malloc(num_bytes);
        int *expected_out = (int*)malloc(num_bytes);
        int *d_in, *d_out;
        dim3 blocks(N/K, N/K);          // blocks per grid
        dim3 threads(K, K);             // threads per block
        dim3 blocks16x16(N/16, N/16);   // blocks per grid
        dim3 threads16x16(16, 16);      // threads per block
        cudaDeviceProp prop;            // CUDA device properties   
        int device = 0;                 // ID of device for GPU execution
        double peakMemBwGbps;           // GPU peak memory bandwidth in Gbps 
        double memUtil;                 // GPU memory bandwidth utilization

        // no enough host memory
        if (!h_in || !h_out || !expected_out) {
                goto out;
        }

        // initialize matrix with random numbers
        fill_matrix(h_in);

        // transpose the matrix and get the expected matrix
        gettimeofday(&start_time, NULL);
        transpose_cpu(h_in, expected_out);
        gettimeofday(&stop_time, NULL);
        elapsed_time = (stop_time.tv_sec - start_time.tv_sec) * 1000 + (stop_time.tv_usec - start_time.tv_usec) / 1000.0;
        printf("CPU time: %f ms\n", elapsed_time);

        // Set device to be used for GPU executions
        checkCudaErrors(cudaSetDevice(device));
        // Get device properties
        cudaGetDeviceProperties(&prop, device);
        // Calculate peark memory bandwidth (GB/s) of GPU
        peakMemBwGbps = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
        printf("Peak memory bandwidth of GPU %d is %f GB/s\n", device, peakMemBwGbps);
        printf("====================================================\n");

        // allocate GPU memory
        checkCudaErrors(cudaMalloc(&d_in, num_bytes));
        checkCudaErrors(cudaMalloc(&d_out, num_bytes));

        // copy input from host memory to GPU memory
        checkCudaErrors(cudaMemcpy(d_in, h_in, num_bytes, cudaMemcpyHostToDevice));

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // launch serial kernel
        cudaEventRecord(start);
        transpose_serial<<<1, 1>>>(d_in, d_out);
        cudaEventRecord(stop);

        // copy output from GPU memory to host memory
        checkCudaErrors(cudaMemcpy(h_out, d_out, num_bytes, cudaMemcpyDeviceToHost));

        // calculate elapsed time in ms and memory utilization
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        memUtil = (2 * N * N * sizeof(int)) / (elapsed_time / 1.0e3) / (peakMemBwGbps * 1.0e9);

        printf("transpose_serial\nTime: %f ms\nMemory utilization %f\%\n%s results\n", 
               elapsed_time,
               memUtil * 100,
               same_matrices(h_out, expected_out) ? "Correct" : "Wrong");
        printf("====================================================\n");

        // launch parallel per row kernel
        cudaEventRecord(start);
        transpose_parallel_per_row<<<1, N>>>(d_in, d_out);        
        cudaEventRecord(stop);

        // copy output from GPU memory to host memory
        checkCudaErrors(cudaMemcpy(h_out, d_out, num_bytes, cudaMemcpyDeviceToHost));

        // calculate elapsed time in ms and memory utilization
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        memUtil = (2 * N * N * sizeof(int)) / (elapsed_time / 1.0e3) / (peakMemBwGbps * 1.0e9);        

        printf("transpose_parallel_per_row\nTime: %f ms\nMemory utilization %f\%\n%s results\n", 
               elapsed_time,
               memUtil * 100,                
               same_matrices(h_out, expected_out) ? "Correct" : "Wrong");
        printf("====================================================\n");

        // launch parallel per element kernel
        cudaEventRecord(start);
        transpose_parallel_per_element<<<blocks, threads>>>(d_in, d_out);        
        cudaEventRecord(stop);

        // copy output from GPU memory to host memory
        checkCudaErrors(cudaMemcpy(h_out, d_out, num_bytes, cudaMemcpyDeviceToHost));

        // calculate elapsed time in ms and check results
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        memUtil = (2 * N * N * sizeof(int)) / (elapsed_time / 1.0e3) / (peakMemBwGbps * 1.0e9); 

        printf("transpose_parallel_per_element\nTime: %f ms\nMemory utilization %f\%\n%s results\n", 
               elapsed_time,
               memUtil * 100, 
               same_matrices(h_out, expected_out) ? "Correct" : "Wrong");
        printf("====================================================\n");

        // launch parallel per element tiled kernel
        cudaEventRecord(start);
        transpose_parallel_per_element_tiled<<<blocks, threads>>>(d_in, d_out);        
        cudaEventRecord(stop);

        // copy output from GPU memory to host memory
        checkCudaErrors(cudaMemcpy(h_out, d_out, num_bytes, cudaMemcpyDeviceToHost));

        // calculate elapsed time in ms and check results
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        memUtil = (2 * N * N * sizeof(int)) / (elapsed_time / 1.0e3) / (peakMemBwGbps * 1.0e9); 

        printf("transpose_parallel_per_element_tiled (block: %d x %d)\nTime: %f ms\nMemory utilization %f\%\n%s results\n", 
               K,
               K,
               elapsed_time,
               memUtil * 100, 
               same_matrices(h_out, expected_out) ? "Correct" : "Wrong");
        printf("====================================================\n");

        // launch parallel per element tiled kernel with different block size (16x16)
        cudaEventRecord(start);
        transpose_parallel_per_element_tiled16<<<blocks16x16, threads16x16>>>(d_in, d_out);        
        cudaEventRecord(stop);

        // copy output from GPU memory to host memory
        checkCudaErrors(cudaMemcpy(h_out, d_out, num_bytes, cudaMemcpyDeviceToHost));

        // calculate elapsed time in ms and check results
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        memUtil = (2 * N * N * sizeof(int)) / (elapsed_time / 1.0e3) / (peakMemBwGbps * 1.0e9); 

        printf("transpose_parallel_per_element_tiled (block: 16 x 16)\nTime: %f ms\nMemory utilization %f\%\n%s results\n", 
               elapsed_time,
               memUtil * 100, 
               same_matrices(h_out, expected_out) ? "Correct" : "Wrong");
        printf("====================================================\n");        

        // free GPU memory
        cudaFree(d_in);
        cudaFree(d_out);

out:
        free(h_in);
        free(h_out);
        free(expected_out);

        return 0;
}