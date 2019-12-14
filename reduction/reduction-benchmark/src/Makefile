BSIZE=256
R=1
DEBUG=DUMMY
PARAMS=-O3 -DBSIZE=${BSIZE} -DR=${R} -D${DEBUG}
ARCH=sm_70
all:
	nvcc ${PARAMS} -rdc=true -arch ${ARCH} main.cu -o prog
old:
	nvcc ${PARAMS} main.cu -o prog
