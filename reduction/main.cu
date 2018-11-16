
#include <cuda.h>
#include <mma.h>
#include <cstdio>
#define REAL float
#define TCSIZE 16
#define TCSQ 256
#define BSIZE 32
#define PRINTLIMIT 10
#include "kernel.cuh"

void initmat(REAL *m, int nmats, const int val){
    for(int k=0; k<nmats; ++k){
        int off = k*TCSIZE*TCSIZE;
        for(int i=0; i<TCSIZE; ++i){
            for(int j=0; j<TCSIZE; ++j){
                m[off + i*TCSIZE + j] = (val*(k+1));
            }
        }
    }
}

void printmats(REAL *m, int nmats, const char *msg){
    printf("%s:\n", msg);
    for(int k=0; k<nmats; ++k){
        printf("k=%i\n", k);
        int off = k*TCSIZE*TCSIZE;
        for(int i=0; i<TCSIZE; ++i){
            for(int j=0; j<TCSIZE; ++j){
                printf("%.2f ", m[off + i*TCSIZE + j]);
            }
            printf("\n");
        }
    }
}

int main(int argc, char **argv){
    // params
    if(argc != 3){
        fprintf(stderr, "run as ./prog dev nmats\n");
        exit(EXIT_FAILURE);
    }
    int dev = atoi(argv[1]);
    int nmats = atoi(argv[2]);
    int totaln = nmats*(TCSIZE)*(TCSIZE);
    printf("nmats=%i  dev=%i   TCSIZE=%i  totaln=%i\n", nmats, dev, TCSIZE, totaln);
    
    // set device
    cudaSetDevice(dev);

    // mallocs
    REAL *A,  *C;
    REAL *Ad, *Cd;
    half *Adh;

    A = (REAL*)malloc(sizeof(REAL)*totaln);
    C = (REAL*)malloc(sizeof(REAL)*totaln);

    cudaMalloc(&Ad, sizeof(REAL)*totaln);
    cudaMalloc(&Cd, sizeof(REAL)*totaln);

    cudaMalloc(&Adh, sizeof(half)*totaln);

    initmat(A, nmats, 1);
    initmat(C, nmats, 0);

    cudaMemcpy(Ad, A, sizeof(REAL)*totaln, cudaMemcpyHostToDevice);
    cudaMemcpy(Cd, C, sizeof(REAL)*totaln, cudaMemcpyHostToDevice);

    convertFp32ToFp16 <<< (totaln + 255)/256, 256 >>> (Adh, Ad, totaln);

    dim3 block, grid;

    block = dim3(TCSIZE, 2, 1);
    grid = dim3(nmats, 1, 1);
    warpReduceSumTC<<<grid, block>>>(Adh, Cd, totaln);
    convertFp32ToFp16 <<< (totaln + 255)/256, 256 >>> (Adh, Cd, totaln);
    warpReduceSumTC<<<grid, block>>>(Adh, Cd, totaln);

    cudaDeviceSynchronize();
  
    cudaMemcpy(A, Ad, sizeof(REAL)*totaln, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, Cd, sizeof(REAL)*totaln, cudaMemcpyDeviceToHost);

    if(nmats < PRINTLIMIT){
        printmats(A, nmats, "[after] mat A:");
        printmats(C, nmats, "[after] mat C:");
    }

    free(A);
    free(C);

    cudaFree(Ad);
    cudaFree(Cd);
    cudaFree(Adh);

    exit(EXIT_SUCCESS);
}

