BSIZE=256
R=1
DEBUG=DUMMY
KDEBUG=DUMMY
PARAMS=-O3 -DBSIZE=${BSIZE} -DR=${R} -D${DEBUG} -D${KDEBUG}
ARCH=sm_75
all:
	nvcc ${PARAMS} -arch ${ARCH} src/main.cu -o prog
