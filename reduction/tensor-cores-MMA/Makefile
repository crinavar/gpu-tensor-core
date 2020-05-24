BSIZE=256
R=1
DEBUG=DUMMY
KDEBUG=DUMMY
NPROC=16
PARAMS=-lnvidia-ml -O3 -DNPROC=${NPROC} -DBSIZE=${BSIZE} -DR=${R} -D${DEBUG} -D${KDEBUG} --default-stream per-thread -Xcompiler -lpthread -Xcompiler -fopenmp
ARCH=sm_75
all:
	nvcc ${PARAMS} -arch ${ARCH} src/main.cu src/nvmlPower.cpp -o bin/prog
