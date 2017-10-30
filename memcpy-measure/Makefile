all: main.cu
	nvcc -I /usr/local/cuda-8.0/samples/common/inc main.cu -o memcpy-measure
clean:
	rm -f memcpy-measure