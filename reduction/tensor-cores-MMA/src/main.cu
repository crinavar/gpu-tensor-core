
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#define REAL float
#define TCSIZE 16
#define TCSQ 256
#define PRINTLIMIT 2560
#define WARPSIZE 32
#define DIFF (BSIZE<<3)

#include "tools.cuh"
#include "kernel.cuh"
#include "variants.cuh"


int main(int argc, char **argv){
    // params
    if(argc != 8){
        fprintf(stderr, "run as ./prog dev n factor_ns seed REPEATS dist method\nmethod:\
        \n0 -> warp-shuffle    \
        \n1 -> recurrence\
        \n2 -> chained MMAs\
        \n3 -> split\
        \n4 -> recursive-chained\n\n");
        exit(EXIT_FAILURE);
    }
    int dev = atoi(argv[1]);
    long on = atoi(argv[2]);
    long n = on;
    float factor_ns = atof(argv[3]);
    int seed = atoi(argv[4]);
    int REPEATS = atoi(argv[5]);
    int dist = atoi(argv[6]);
    int method = atoi(argv[7]);

#ifdef DEBUG
    const char* methods[5] = {"WARP-SHUFFLE", "RECURRENCE", "CHAINED MMAs", "SPLIT", "RECURRENCE-CHAINED"};
    const char* disttext[3] = {"Normal Distribution", "Uniform Distribution", "Constant Distribution"};
    printf("\n\
            ***************************\n\
            dev            = %i\n\
            method         = %s\n\
            n              = %i\n\
            factor_ns      = %f\n\
            dist           = %s\n\
            prng_seed      = %i\n\
            REPEATS        = %i\n\
            TCSIZE         = %i\n\
            R              = %i\n\
            BSIZE          = %i\n\
            ***************************\n\n", dev, methods[method], n, factor_ns, disttext[dist], seed, REPEATS, TCSIZE, R, BSIZE);
#endif
    
    // set device
    cudaSetDevice(dev);

    // mallocs
    REAL *A, *Ad;
    half *Adh, *outd_recA, *outd_recB;
    float *outd, *out;

    A = (REAL*)malloc(sizeof(REAL)*n);
    out = (float*)malloc(sizeof(float)*1);
    cudaMalloc(&Ad, sizeof(REAL)*n);
    cudaMalloc(&Adh, sizeof(half)*n);
    cudaMalloc(&outd, sizeof(float)*1);
    long smalln = (n + TCSQ-1)/TCSQ;
    //printf("small n = %lu   bs = %i\n", smalln, bs);
    cudaMalloc(&outd_recA, sizeof(half)*(smalln));
    cudaMalloc(&outd_recB, sizeof(half)*(smalln));

    init_distribution(A, n, seed, dist);
    cudaMemcpy(Ad, A, sizeof(REAL)*n, cudaMemcpyHostToDevice);
    convertFp32ToFp16 <<< (n + 256 - 1)/256, 256 >>> (Adh, Ad, n);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    #ifdef DEBUG
        printf("%s (BSIZE = %i)\n", methods[method], BSIZE);
    #endif
    cudaEventRecord(start);
    switch(method){
        case 0:
            warpshuffle_reduction(Adh, outd, n, REPEATS);
            break;
        case 1:
            recurrence_reduction(Adh, outd, outd_recA, outd_recB, n, REPEATS);
            break;
        case 2:
            chainedMMAs_reduction(Adh, outd, n, REPEATS);
            break;
        case 3:
            split_reduction(Adh, outd, n, factor_ns, REPEATS);
            break;
        case 4:
            recurrence_reduction_chained(Adh, outd, outd_recA, outd_recB, n, REPEATS);
            break;
    }        
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(out, outd, sizeof(float)*1, cudaMemcpyDeviceToHost);
    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    double cpusum = gold_reduction(A, n);

    #ifdef DEBUG
        printf("Done:\n\
                Time (GPU):  %f\n\
                GPU Result:  %f\n\
                CPU Result:  %f\n\
                Diff Result: %f\n\
                \% Error   : %f\n\n", time/(REPEATS),(float)*out,cpusum,fabs((float)*out - cpusum),fabs(100.0f*fabs((float)*out - cpusum)/cpusum));
    #else
        printf("%f,%f,%f,%f,%f\n", time/(REPEATS),(float)*out,cpusum,fabs((float)*out - cpusum),fabs(100.0f*fabs((float)*out - cpusum)/cpusum));
    #endif
    free(A);
    free(out);
    cudaFree(Ad);
    cudaFree(Adh);
    cudaFree(outd);
    cudaFree(outd_recA);
    cudaFree(outd_recB);
    exit(EXIT_SUCCESS);
}

