
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#define REAL float
#define TCSIZE 16
#define TCSQ 256
#define BSIZE 32
#define PRINTLIMIT 2560
#define REPEATS 10000
#define WARPSIZE 32
#include "kernel.cuh"

void init(REAL *m, long n, const int val, int seed){
    srand(seed);
    for(long k=0; k<n; ++k){
        //m[k] = 0.001f;
        //m[k] = (REAL) rand()/((REAL)RAND_MAX)
        m[k] = 1.000f;
    }
}

void printarray(REAL *m, int n, const char *msg){
    printf("%s:\n", msg);
    for(int j=0; j<n; ++j){
        printf("%8.4f\n", m[j]);
    }
}

void printmats(REAL *m, int n, const char *msg){
    long nmats = (n + 256 - 1)/TCSQ;
    printf("%s:\n", msg);
    long index=0;
    for(int k=0; k<nmats; ++k){
        printf("k=%i\n", k);
        int off = k*TCSIZE*TCSIZE;
        for(int i=0; i<TCSIZE; ++i){
            for(int j=0; j<TCSIZE; ++j){
                if(index < n){
                    printf("%8.4f ", m[off + i*TCSIZE + j]);
                }
                else{
                    printf("%8.4f ", -1.0f);
                }
                index += 1;
            }
            printf("\n");
        }
    }
}

int main(int argc, char **argv){
    // params
    if(argc != 4){
        fprintf(stderr, "run as ./prog dev n seed\n");
        exit(EXIT_FAILURE);
    }
    int dev = atoi(argv[1]);
    int on = atoi(argv[2]);
    int n = on;
    int seed = atoi(argv[3]);
    printf("n=%i dev=%i  rand_seed = %i  TCSIZE=%i\n", n, dev, TCSIZE);
    
    // set device
    cudaSetDevice(dev);

    // mallocs
    REAL *A;
    REAL *Ad;
    half *Adh;
    float *outd;
    float *out;

    A = (REAL*)malloc(sizeof(REAL)*n);
    out = (float*)malloc(sizeof(float)*1);
    cudaMalloc(&Ad, sizeof(REAL)*n);
    cudaMalloc(&Adh, sizeof(half)*n);
    cudaMalloc(&outd, sizeof(float)*1);

    init(A, n, 1, seed);
    //printmats(A, n, "[after] mat A:");
    cudaMemcpy(Ad, A, sizeof(REAL)*n, cudaMemcpyHostToDevice);
    convertFp32ToFp16 <<< (n + 256 - 1)/256, 256 >>> (Adh, Ad, n);
    cudaDeviceSynchronize();
    
    //printmats(A, n, "[after] mat A:");
    
    dim3 block, grid;
    //block = dim3(TCSIZE*2, 1, 1);
    //grid = dim3((n + 256 - 1)/TCSQ, 1, 1);
    block = dim3(TCSIZE*2*32, 1, 1);
    grid = dim3((n + (TCSQ*32) - 1)/(TCSQ*32), 1, 1);
    printf("grid (%i, %i, %i)    block(%i, %i, %i)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
   
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    //for(int i=0; i<REPEATS; ++i){
        kernel_reduction<<<grid, block>>>(Adh, outd, n);
        cudaDeviceSynchronize();
        /*
        while(n > 1){
            tc_reduction<<<grid, block>>>(Adh, n);
            cudaDeviceSynchronize();
            // actualizar n --> nuevo n 'paddeado' y mas chico
            n = ((n + 255)/TCSQ) * 256;
            n = n/256;
            //n = ((n + 255) >> 256);
            // n/TCSQ
            grid = dim3((n + 255)/TCSQ, 1, 1);
            //printf("n: %i, grid: %i, %i, %i\n",n,grid.x,grid.y,grid.z);
            //printmats(A, n, "[after] mat D:");
        }
        n=on;
        grid = dim3((n + 256 - 1)/TCSQ, 1, 1);
        */
    //}
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf("time: %f \n", time/1000.0f);
    //printf("%f \n", time/(REPEATS*1000.0f));

    convertFp16ToFp32 <<< (n + 255)/256, 256 >>> (Ad, Adh, n);
    cudaMemcpy(A, Ad, sizeof(REAL)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(out, outd, sizeof(float)*1, cudaMemcpyDeviceToHost);
    
    printf("sum: %f \n", *out);
    /*if(n < PRINTLIMIT){
       printmats(A, on, "A final:");
    }*/

    //printarray(A, 1, "D_final: ");

    free(A);
    free(out);
    cudaFree(Ad);
    cudaFree(Adh);
    cudaFree(outd);
    exit(EXIT_SUCCESS);
}

