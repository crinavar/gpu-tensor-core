ARCH=sm_70
POWER=DUMMY
OPTIONS=-lnvidia-ml -D${POWER} -Xcompiler -lpthread,-fopenmp
SRCFLOAT=$(wildcard *.cpp) 	cub_reduce_float.cu
SRCHALF=$(wildcard *.cpp) 	cub_reduce_half.cu
all:
	nvcc -arch=${ARCH} ${OPTIONS} ${SRCHALF} -o prog_cub16 -I../.. -lcudart -O3
	nvcc -arch=${ARCH} ${OPTIONS} ${SRCFLOAT} -o prog_cub32 -I../.. -lcudart -O3
half: 
	nvcc -arch=${ARCH} ${OPTIONS} ${SRCHALF} -o prog_cub16 -I../.. -lcudart -O3
float:
	nvcc -arch=${ARCH} ${OPTIONS} ${SRCFLOAT} -o prog_cub32 -I../.. -lcudart -O3
