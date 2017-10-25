all: main.cu
	nvcc -I /usr/local/cuda-8.0/samples/common/inc main.cu -o kernel-launch-measure
clean:
	rm -f kernel-launch-measure