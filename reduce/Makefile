all: reduce.cu
	nvcc -arch=compute_60 -code=sm_60 -I /usr/local/cuda-8.0/samples/common/inc -o reduce reduce.cu
clean:
	rm -f reduce