#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mma.h>
#define BSIZE 512
#define WARPSIZE 32
#define REAL float
#define REPEATS 1000
#include "reduction.cuh"

int main(int argc, char **argv){
	if(argc != 3){
		fprintf(stderr, "run as ./prog dev n \n ");
		exit(EXIT_FAILURE);
	}
	REAL *a, *ad, *out, *outd;
    int dev = atoi(argv[1]);
    int n = atoi(argv[2]);
    cudaSetDevice(dev);
	a = (REAL*)malloc(sizeof(REAL)*n);
	out = (REAL*)malloc(sizeof(REAL)*1);
	*out = 0.0f;
	for(int i=0; i<n; ++i){
		a[i] = 0.001f;//(REAL)rand()/RAND_MAX;
		//a[i] = 1;//(REAL)rand()/RAND_MAX;
	}

	cudaMalloc(&ad, sizeof(REAL)*n);
	cudaMalloc(&outd, sizeof(REAL)*1);
	cudaMemcpy(ad, a, sizeof(REAL)*n, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

	dim3 block(BSIZE, 1, 1);
	dim3 grid((n + BSIZE - 1)/BSIZE, 1, 1);
	printf("grid(%i, %i, %i),  block(%i, %i, %i)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    for(int i=0; i < REPEATS; ++i){
        kernel_reduction<REAL><<<grid, block>>>(ad, n, outd);
        //deviceReduceKernel<REAL><<<grid, block>>>(ad, outd, n);
        //deviceReduceKernel<REAL><<<1,1024>>>(outd, outd, (n + BSIZE - 1)/BSIZE);
        
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
 
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf("%f \n", time/(REPEATS * 1000.0f));
    
    cudaMemcpy(out, outd, sizeof(REAL)*1, cudaMemcpyDeviceToHost);
	printf("[GPU] result = %f\n", (float) *out);
}

