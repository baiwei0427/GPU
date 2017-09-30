#include <stdio.h>

const int N = 16;     // matrix size is N x N
//const int K = 4;        // tile size is K x K 

void transpose_cpu(int *in, int *out) {
        for (int row = 0; row < N; row++) {
                for (int col = 0; col < N; col++) {
                        out[col * N + row] = in[row * N + col];
                }
        }
}

void print_matrix(int *in) {
        for (int row = 0; row < N; row++) {
                for (int col = 0; col < N; col++) {
                        printf("%d ", in[row * N + col]);
                }
                printf("\n");
        }
}

void fill_matrix(int *in) {
        int size = N * N;
        for (int i = 0; i < size; i++) {
                in[i] = rand() % 10;
        }
}

int main(int argc, char **argv) 
{
        int *h_in = (int*)malloc(N * N * sizeof(int));
        int *h_out = (int*)malloc(N * N * sizeof(int));

        // no enough host memory
        if (!h_in || !h_out) {
                goto out;
        }

        // initialize matrix with random numbers
        fill_matrix(h_in);

        // print input matrix
        printf("Input\n");
        print_matrix(h_in);
        
        // transpose the matrix
        transpose_cpu(h_in, h_out);

        // print output matrix
        printf("\nOutput\n");
        print_matrix(h_out);

out:
        free(h_in);
        free(h_out);

        return 0;
}