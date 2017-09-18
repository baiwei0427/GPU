#include <stdio.h>

void merge(int *array, int start, int mid, int end)
{       
        int left_index, right_index, global_index;
        int left_len = mid - start + 1;
        int right_len = end - mid;
        int left[left_len];
        int right[right_len];

        // initialize left array
        for (int i = 0; i < left_len; i++) {
                left[i] = array[start + i];
        }

        // initialize right array
        for (int i = 0; i < right_len; i++) {
                right[i] = array[mid + 1 + i];
        }

        // index of left array
        left_index = 0;
        // index of right array
        right_index = 0;
        // index of merged array
        global_index = start;

        while (left_index < left_len && right_index < right_len) {
                if (left[left_index] <= right[right_index]) {
                        array[global_index++] = left[left_index++];
                } else {
                        array[global_index++] = right[right_index++];                        
                }
        }

        // copy the rest of left array 
        while (left_index < left_len) {
                array[global_index++] = left[left_index++];
        } 

        // copy the rest of right array
        while (right_index < right_len) {
                array[global_index++] = right[right_index++];
        }
}

void cpu_merge_sort(int *array, int start, int end)
{
        if (start >= end)
                return;
        
        int mid = start + (end - start) / 2;
        cpu_merge_sort(array, start, mid);
        cpu_merge_sort(array, mid + 1, end);
        merge(array, start, mid, end);
}

int main()
{
        int array_size = 1000;
        int array[array_size];
        
        // initialize random number generator
        srand(time(NULL));

        printf("Input\n");
        for (int i = 0; i < array_size; i++) {
                array[i] = rand() % array_size;
                printf("%d ", array[i]);
        }
        printf("\n");

        cpu_merge_sort(array, 0, array_size - 1);

        printf("Output\n"); 
        for (int i = 0; i < array_size; i++) {
                printf("%d ", array[i]);
        }        
        printf("\n");
        return 0;
}