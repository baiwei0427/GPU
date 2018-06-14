all: main.cu
	nvcc -I /usr/local/cuda-8.0/samples/common/inc main.cu -o sync-overhead-measure
clean:
	rm -f sync-overhead-measure