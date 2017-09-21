#include <stdio.h>

inline bool is_sorted_array(unsigned int *array, int array_size)
{        
        for (int i = 1; i < array_size; i++) {
                if (array[i] < array[i - 1]) {
                        return false;
                }
        }
        return true;
}

// get the maximum value of the array
inline unsigned int get_max(unsigned int *array, int array_size)
{       
        // min of unsigned int is 0
        unsigned int result = 0U;

        for (int i = 0; i < array_size; i++) {
                if (array[i] > result) {
                        result = array[i];
                }
        }

        return result;
}

void print_array(unsigned int *array, int array_size) {
        for (int i = 0; i < array_size; i++) {
                printf("%u ", array[i]);
        }
        printf("\n");
}

void count_sort(unsigned int *array, int array_size, unsigned int radix, unsigned int exp)
{
        unsigned int *output = NULL;
        unsigned int count[radix], digit;

        memset(count, 0, sizeof(unsigned int) * radix);
        if (!(output = (unsigned int*)malloc(array_size * sizeof(unsigned int))))
                return;

        // store count of occurrences in count[]
        for (int i = 0; i < array_size; i++) {
                digit = (array[i] / exp) % radix;
                count[digit]++;
        }
        
        // run inclusive scan for count
        for (int i = 1; i < radix; i++) {
                count[i] += count[i - 1];
        }

        // build the output array
        for (int i = array_size - 1; i >= 0; i--) {
                digit = (array[i] / exp) % radix;
                output[count[digit] - 1] = array[i];
                count[digit]--;
        }

        // copyt output to array
        for (int i = 0; i < array_size; i++) {
                array[i] = output[i];
        }

        free(output);
}

void radix_sort(unsigned int *array, int array_size, unsigned int radix)
{
        if (array_size <= 1 || radix <= 1) {
                return;
        }

        // find the maximum number in this array
        unsigned int max_val = get_max(array, array_size);

        // Do counting sort for every digit. Note that instead
        // of passing digit number, exp is passed. exp is radix ^ i
        // where i is current digit number
        for (unsigned int exp = 1; exp <= max_val; exp *= radix) {
                count_sort(array, array_size, radix, exp);
        }
}

int main()
{
        int array_size = 1024;
        unsigned int array[array_size];

        // initialize random number generator
        srand(time(NULL));

        // initialize input
        for (int i = 0; i < array_size; i++) {
                array[i] = rand() % array_size;
        }

        // sort the array
        radix_sort(array, array_size, 10);

        print_array(array, array_size);

        // check sorting results
        if (is_sorted_array(array, array_size)) {
                printf("Correct\n");
        } else {
                printf("Wrong\n");
        }

        return 0;
}