ARCH=sm_70
all:
	nvcc -arch=${ARCH} cub_reduce_half.cu -o prog_cub16 -I../.. -lcudart -O3
half: 
	nvcc -arch=${ARCH} cub_reduce_half.cu -o prog_cub16 -I../.. -lcudart -O3
float:
	nvcc -arch=${ARCH} cub_reduce_float.cu -o prog_cub32 -I../.. -lcudart -O3
