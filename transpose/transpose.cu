#include <stdio.h>
#include <sys/time.h>
#include "helper_cuda.h"

const int N = 1024;     // matrix size is N x N
//const int K = 32;       // tile size is K x K 

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

        // calculate elapsed time in ms and check results
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("transpose_serial time: %f ms\n%s results\n", elapsed_time, 
               same_matrices(h_out, expected_out) ? "Correct" : "Wrong");

        // free GPU memory
        cudaFree(d_in);
        cudaFree(d_out);

out:
        free(h_in);
        free(h_out);

        return 0;
}