
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#define REAL float
#define TCSIZE 16
#define TCSQ 256
#define PRINTLIMIT 2560
#define WARPSIZE 32
#define DIFF (BSIZE<<3)
#include "kernel.cuh"

void init(REAL *m, long n, const int val, int seed){
    srand(seed);
    for(long k=0; k<n; ++k){
        //m[k] = 0.001f;
        //m[k] = (REAL) rand()/((REAL)RAND_MAX);
        m[k] = (REAL) rand()/(((REAL)RAND_MAX)*100);
        //m[k] = 0.001f;
    }
}

float gold_reduction(REAL *m, long n){
    float sum = 0.0f;
    for(long k=0; k<n; ++k){
        sum += m[k];
    }
    return sum;
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
    if(argc != 7){
        fprintf(stderr, "run as ./prog dev n factor_ns seed REPEATS method\n");
        exit(EXIT_FAILURE);
    }
    int dev = atoi(argv[1]);
    int on = atoi(argv[2]);
    int n = on;
    float factor_ns = atof(argv[3]);
    int seed = atoi(argv[4]);
    int REPEATS = atoi(argv[5]);
    int method = atoi(argv[6]);
#ifdef DEBUG
    printf("n=%i factor_ns=%i dev=%i  rand_seed = %i  REPEATS = %i  TCSIZE=%i method=%i\n", n, factor_ns, dev, seed, REPEATS, TCSIZE, method);
#endif
    
    // set device
    cudaSetDevice(dev);

    // mallocs
    half z = 0.01;
    REAL *A;
    REAL *Ad;
    half *Adh;
    float *outd;
    half *outd_m0;
    float *out;

    A = (REAL*)malloc(sizeof(REAL)*n);
    out = (float*)malloc(sizeof(float)*1);
    cudaMalloc(&Ad, sizeof(REAL)*n);
    cudaMalloc(&Adh, sizeof(half)*n);
    cudaMalloc(&outd, sizeof(float)*1);
    cudaMalloc(&outd_m0, sizeof(half)*n);

    init(A, n, 1, seed);
    //printmats(A, n, "[after] mat A:");
    cudaMemcpy(Ad, A, sizeof(REAL)*n, cudaMemcpyHostToDevice);
    convertFp32ToFp16 <<< (n + 256 - 1)/256, 256 >>> (Adh, Ad, n);
    cudaDeviceSynchronize();
    
    //printmats(A, n, "[after] mat A:");
    
    dim3 block, grid;
    //block = dim3(TCSIZE*2, 1, 1);
    //grid = dim3((n + 256 - 1)/TCSQ, 1, 1);
    int bs = BSIZE/(TCSIZE*2);
    
    //block = dim3(TCSIZE*2*bs, 1, 1);
    //grid = dim3((n + (TCSQ*bs*(R)) - 1)/(TCSQ*bs*(R)), 1, 1);
    //printf("grid (%i, %i, %i)    block(%i, %i, %i)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
   
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    if(method == 1){
        if(n<=524288){
        //printf("reduction_tc_theory (cooperative groups)\n");
        void *kernelArgs[3];
        kernelArgs[0]= (void*)&Adh;
        kernelArgs[1]= (void*)&outd_m0;
        kernelArgs[2]= (void*)&n;
        block = dim3(BSIZE, 1, 1);
        grid = dim3((n + DIFF -1)/(DIFF),1,1) ;//dim3((n + 255)/TCSQ, 1, 1);
        //printf("grid (%i, %i, %i)    block(%i, %i, %i)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
        //kernel_reduction_tc_theory<<<grid, block>>>(Adh, outd_m0, n);
        for(int i=0; i<REPEATS; ++i){
            cudaMemset(outd_m0, 0, sizeof(REAL)*1);
            cudaLaunchCooperativeKernel((void *) kernel_reduction_tc_theory,grid,block,kernelArgs,NULL);  CUERR;
            cudaDeviceSynchronize();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaMemcpy(&z, outd_m0, sizeof(half)*1, cudaMemcpyDeviceToHost);
        *out = (float)z;
        //cudaMemcpy(out, outd, sizeof(half)*1, cudaMemcpyDeviceToHost);
        /*n = ((n + TCSQ-1)/TCSQ) * TCSQ;
        grid = dim3((n + TCSQ-1)/TCSQ, 1, 1);
        while(n > 1){
            kernel_reduction_tc_theory<<<grid, block>>>(outd_m0, outd_m0, n);
            // para n generico: actualizar n --> nuevo n 'paddeado' y mas chico
            n = ((n + TCSQ-1)/TCSQ) * TCSQ;
            // para n potencias de TCSQ 
            //n = n/256;
            // n/TCSQ
            grid = dim3((n + TCSQ-1)/TCSQ, 1, 1);
            //printf("grid (%i, %i, %i)    block(%i, %i, %i)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
            //printf("n: %i, grid: %i, %i, %i\n",n,grid.x,grid.y,grid.z);
            //printmats(A, n, "[after] mat D:");
        }
        n=on;
        //grid = dim3((n + 256 - 1)/TCSQ, 1, 1);
        //cudaMemcpy(Ad, A, sizeof(REAL)*n, cudaMemcpyDeviceToHost);    
        //printf("D: %f\n",(float)A[0]);
    */
        }
        else{
            printf("0,0,0,0,0\n");
            free(A);
            free(out);
            cudaFree(Ad);
            cudaFree(Adh);
            cudaFree(outd);
            //*out = 0.0f;
            exit(EXIT_SUCCESS);
        }
    }
    if(method == 2){
        //printf("reduction_tc_blockshuffle\n");
        block = dim3(TCSIZE*2*bs, 1, 1);
        grid = dim3((n + (TCSQ*bs*(R)) - 1)/(TCSQ*bs*(R)), 1, 1);
        //printf("grid (%i, %i, %i)    block(%i, %i, %i)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
        for(int i=0; i<REPEATS; ++i){
            cudaMemset(outd, 0, sizeof(REAL)*1);
            kernel_reduction_tc_blockshuffle<<<grid, block>>>(Adh, outd, n,bs);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaMemcpy(out, outd, sizeof(float)*1, cudaMemcpyDeviceToHost);
    }
    if(method == 3){
        //printf("reduction_tc_mixed\n");
        block = dim3(BSIZE, 1, 1);
        int ns_blocks = (factor_ns*n + BSIZE-1)/BSIZE;
        //printf("s_block %i\n",ns_blocks);
        float tc_blocks = (n*(1-factor_ns) + DIFF -1)/(DIFF);
        //float tc_blocks = n*0.5/(DIFF);
        grid = dim3((int)tc_blocks + ns_blocks, 1, 1);
        //printf("grid (%i, %i, %i)    block(%i, %i, %i)  DIFF %i\n", grid.x, grid.y, grid.z, block.x, block.y, block.z,DIFF);
        //printf("ns_blocks %i, tc_blocks %i\n",ns_blocks,(int)tc_blocks);
        for(int i=0; i<REPEATS; ++i){
            cudaMemset(outd, 0, sizeof(REAL)*1);
            kernel_reduction_tc_mixed<<<grid, block>>>(Adh, outd, tc_blocks, ns_blocks);  CUERR;
            cudaDeviceSynchronize();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaMemcpy(out, outd, sizeof(float)*1, cudaMemcpyDeviceToHost);
    }        
    if(method == 0){
        block = dim3(BSIZE, 1, 1);
        grid = dim3((n + BSIZE -1)/BSIZE, 1, 1);
        for(int i=0; i<REPEATS; ++i){
            cudaMemset(outd, 0, sizeof(REAL)*1);
            kernel_reduction_shuffle<<<grid, block>>>(Adh, outd, n);  CUERR;
            cudaDeviceSynchronize();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaMemcpy(out, outd, sizeof(float)*1, cudaMemcpyDeviceToHost);
    }
    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    float cpusum = gold_reduction(A, n);
    //printf("gpu: %f\ncpu: %f \ndiff = %f (%f%%)\n", (float)*out, cpusum, fabs((float)*out - cpusum), 100.0f*fabs((float)*out - cpusum)/cpusum);
    /*/printf("%f \n",(n/(time/1000.0f))/1000000000.0f);
    printf("%f\n", time/(REPEATS));*/
    printf("%f,%f,%f,%f,%f\n",time/(REPEATS),(float)*out,cpusum,fabs((float)*out - cpusum),100.0f*fabs((float)*out - cpusum)/cpusum);
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

