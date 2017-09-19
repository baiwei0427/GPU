#include <stdio.h>
#include <math.h>

int max_power_of_two_less_than(int x)
{
        int result = 1;
        while (result < x) {
                result <<= 1;
        }

        return result >> 1;
}

void compare_swap(int *array, int a, int b, bool up)
{
        if (up == (array[a] > array[b])) {
                int tmp = array[a];
                array[a] = array[b];
                array[b] = tmp;
        }
}

void bitonic_merge(int *array, int start, int size, bool up)
{
        if (size <= 1)
                return;
        
        int m = max_power_of_two_less_than(size);
        for (int i = start; i < start + size - m; i++) {
                compare_swap(array, i, i + m, up);
        }

        // we have two bitonic subarraies now
        // max of the first one <= min of the second one
        // continue the merge recursively  
        bitonic_merge(array, start, m, up);
        bitonic_merge(array, start + m, size - m, up);
}

void bitonic_sort(int *array, int start, int size, bool up)
{
        if (size <= 1)
                return;

        // sort the first subarray in the reverse direction
        bitonic_sort(array, start, size / 2, !up);
        // sort the second subarray in the same direction
        bitonic_sort(array, start + size / 2, size - size / 2, up);

        // we have a bitonic array now, change/merge it into a sorted one
        bitonic_merge(array, start, size, up);
}

void sort(int *array, int size, bool up)
{
        bitonic_sort(array, 0, size, up);
}

int main()
{
        int array_size = 1234;
        int array[array_size];
        
        // initialize random number generator
        srand(time(NULL));

        printf("Input\n");
        for (int i = 0; i < array_size; i++) {
                array[i] = rand() % array_size;
                printf("%d ", array[i]);
        }
        printf("\n");

        sort(array, array_size, true);

        printf("Output\n");
        for (int i = 0; i < array_size; i++) {
                printf("%d ", array[i]);
                if (i >= 1 && array[i] < array[i - 1]) {
                        printf("Wrong\n");
                }
        }         
        printf("\n");  

        return 0;
}