BSIZE=256
R=1
PARAMS=-O3 -DBSIZE=${BSIZE} -DR=${R}
ARCH=sm_70
all:
	nvcc ${PARAMS} -rdc=true -arch ${ARCH} main.cu -o prog
old:
	nvcc ${PARAMS} main.cu -o prog
