BSIZE=256
R=1
DEBUG=DUMMY
PARAMS=-O3 -DBSIZE=${BSIZE} -DR=${R} -D${DEBUG}
ARCH=sm_75
all:
	nvcc ${PARAMS} -arch ${ARCH} src/main.cu -o prog
