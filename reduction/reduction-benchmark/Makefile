BSIZE=32
R=1
PARAMS=-O3 -DBSIZE=${BSIZE} -DR=${R}
all:
	nvcc ${PARAMS} -rdc=true -arch sm_70 main.cu -o prog
old:
	nvcc ${PARAMS} main.cu -o prog
