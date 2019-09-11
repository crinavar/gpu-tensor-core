#include <cuda.h>
#include <mma.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#define REAL float
#define TCSIZE 16
#define TCSQ 256
#define BSIZE 32
#define PRINTLIMIT 10
#include "kernel.cuh"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void last_cuda_error(const char *msg){
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		// print the CUDA error message and exit
		printf("[%s]: CUDA error: %s\n", msg, cudaGetErrorString(error));
		exit(-1);
	}
}

void initmat(REAL *m, int nmats, const int val){
    std::mt19937 gen{12};
    std::normal_distribution<> d(0, 1);
    for(int k=0; k<nmats; ++k){
        int off = k*TCSIZE*TCSIZE;
        for(int i=0; i<TCSIZE; ++i){
            for(int j=0; j<TCSIZE; ++j){
                //m[off + i*TCSIZE + j] = 1;//(j+i*16)/1000;//(val*(k+1));
                m[off + i*TCSIZE + j] = (float) d(gen);
            }
        }
    }
}
void initiden(REAL *m, int nmats, const int val){
    for(int k=0; k<nmats; ++k){
        int off = k*TCSIZE*TCSIZE;
        for(int i=0; i<TCSIZE; ++i){
            for(int j=0; j<TCSIZE; ++j){
                if(i==j){
                    m[off + i*TCSIZE + j] = 1;//(val*(k+1));
                }
                else
                m[off + i*TCSIZE + j] = 0;//(float) d(gen);
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

void timer_start(cudaEvent_t *start, cudaEvent_t *stop){
    cudaEventRecord(*start);
}

void timer_stop(cudaEvent_t *start, cudaEvent_t *stop, const char *msg){
    cudaEventRecord(*stop);
    cudaEventSynchronize(*stop);
    float time = 0.0f;
    cudaEventElapsedTime(&time, *start, *stop);
    printf("%s: %f secs\n", msg, time/1000.0f);
}


// main
int main(int argc, char **argv){
    // params
    if(argc != 4){
        fprintf(stderr, "run as ./prog dev nmats alg\n");
        exit(EXIT_FAILURE);
    }
    int dev = atoi(argv[1]);
    int nmats = atoi(argv[2]);
    int alg = atoi(argv[3]);
    int totaln = nmats*(TCSIZE)*(TCSIZE);
    printf("alg = %i  nmats=%i  dev=%i   TCSIZE=%i  totaln=%i\n", alg, nmats, dev, TCSIZE, totaln);
    
    // set device
    cudaSetDevice(dev);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // mallocs
    REAL *A,  *B,  *C;
    REAL *Ad, *Bd, *Cd;
    half *Adh, *Bdh, *Cdh;

    printf("CPU mallocs (A[], B[], C[])......."); fflush(stdout);
    timer_start(&start, &stop);
    A = (REAL*)malloc(sizeof(REAL)*totaln);
    B = (REAL*)malloc(sizeof(REAL)*totaln);
    C = (REAL*)malloc(sizeof(REAL)*totaln);
    timer_stop(&start, &stop, "done");

    printf("GPU mallocs (A[], B[], C[])......."); fflush(stdout);
    timer_start(&start, &stop);
    cudaMalloc(&Ad, sizeof(REAL)*totaln);
    cudaMalloc(&Bd, sizeof(REAL)*totaln);
    cudaMalloc(&Cd, sizeof(REAL)*totaln);
    cudaMalloc(&Adh, sizeof(half)*totaln);
    cudaMalloc(&Bdh, sizeof(half)*totaln);
    cudaMalloc(&Cdh, sizeof(half)*totaln);
    timer_stop(&start, &stop, "done");

    // init data 
    printf("CPU init (A[], B[], C[]).........."); fflush(stdout);
    timer_start(&start, &stop);
    initiden(A, nmats, 1);
    initmat(B, nmats, 1);
    initmat(C, nmats, 0);
    timer_stop(&start, &stop, "done");
    

    // copy data to GPU
    printf("cudaMemcpy: Host -> Device........"); fflush(stdout);
    timer_start(&start, &stop);
    cudaMemcpy(Ad, A, sizeof(REAL)*totaln, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, sizeof(REAL)*totaln, cudaMemcpyHostToDevice);
    cudaMemcpy(Cd, C, sizeof(REAL)*totaln, cudaMemcpyHostToDevice);
    timer_stop(&start, &stop, "done");

    // convert to half
    convertFp32ToFp16 <<< (totaln + 255)/256, 256 >>> (Adh, Ad, totaln);
    convertFp32ToFp16 <<< (totaln + 255)/256, 256 >>> (Bdh, Bd, totaln);
    convertFp32ToFp16 <<< (totaln + 255)/256, 256 >>> (Cdh, Cd, totaln);

    // parallel structures
    dim3 block, grid;

    // kernel
    if(alg == 1){
        printf("\033[32;1m[matmul sm]\033[0m.................");
        /*block = dim3(TCSIZE, TCSIZE, 1);    
        grid = dim3((totaln+TCSQ-1)/TCSQ, 1, 1);    
        timer_start(&start, &stop);
        matmuls_sm<<<grid, block>>>(Ad, Bd, Cd, totaln);*/
    }
    else if(alg == 2){
        printf("\033[32;1m[matmul tc 16x2-blocks]\033[0m...........");
        block = dim3(TCSIZE, 2, 1);    
        //grid = dim3((totaln+TCSQ-1)/TCSQ, 1, 1);    
        grid = dim3(nmats, 1, 1);    
        timer_start(&start, &stop);
        matmuls_tc<<<grid, block>>>(Adh, Bdh, Cd, totaln);
    }
   
    else{
        printf("error: choose 0..6 for method\n");
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    timer_stop(&start, &stop, "done");
    printf("block(%i, %i, %i) grid(%i, %i, %i)\n", block.x, block.y, block.z, grid.x, grid.y, grid.z); fflush(stdout);


    // copy GPU -> CPU
    printf("cudaMemcpy: Device -> Host........"); fflush(stdout);
    timer_start(&start, &stop);
    cudaMemcpy(A, Ad, sizeof(REAL)*totaln, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, Bd, sizeof(REAL)*totaln, cudaMemcpyDeviceToHost);
    cudaMemcpy(C, Cd, sizeof(REAL)*totaln, cudaMemcpyDeviceToHost);
    timer_stop(&start, &stop, "done");

    if(nmats < PRINTLIMIT){
        printmats(A, nmats, "[after] mat A:");
        printmats(B, nmats, "[after] mat B:");
        printmats(C, nmats, "[after] mat C:");
    }

    // free memory in CPU and GPU
    printf("host free (A, B, C)..............."); fflush(stdout);
    timer_start(&start, &stop);
    free(A);
    free(B);
    free(C);
    timer_stop(&start, &stop, "done");
    printf("device free (A, B, C)............."); fflush(stdout);
    timer_start(&start, &stop);
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
    cudaFree(Adh);
    cudaFree(Bdh);
    timer_stop(&start, &stop, "done");

    // timer results
    exit(EXIT_SUCCESS);
}
