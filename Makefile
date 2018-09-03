PARAMS=-O3
all:
	nvcc ${PARAMS} main.cu -o prog
pascal:
	nvcc ${PARAMS} -arch sm_61 main.cu -o prog
volta:
	nvcc ${PARAMS} -arch sm_70 main.cu -o prog
