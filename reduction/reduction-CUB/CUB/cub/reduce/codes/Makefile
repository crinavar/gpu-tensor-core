all:
	nvcc -arch=sm_70 cub_reduce_half.cu -o prog_cub16 -I../.. -lcudart -O3
half: 
	nvcc -arch=sm_70 cub_reduce_half.cu -o prog_cub16 -I../.. -lcudart -O3
float:
	nvcc -arch=sm_70 cub_reduce_float.cu -o prog_cub32 -I../.. -lcudart -O3
