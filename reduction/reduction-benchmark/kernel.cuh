#include <cooperative_groups.h>
#ifndef _KERNEL_H
#define _KERNEL_H_
using namespace nvcuda;
using namespace cooperative_groups;
//using namespace cooperative_groups;
// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

#define CUERR {                                                            \
    cudaError_t err;                                                       \
    if ((err = cudaGetLastError()) != cudaSuccess) {                       \
        printf("0, 0, 0, 0, 0\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1);                                                           \
    }                                                                      \
}
//printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}
__global__ void convertFp16ToFp32 (float *out, half *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}
__inline__ __device__ REAL shuffle_reduction(REAL val){
	for (int offset = WARPSIZE >> 1; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFF, val, offset, WARPSIZE);
    return val;
}

// kernel
//__global__ void tc_reduction(half* A, int n){
__inline__ __device__ REAL reduction_tc_warp(half *A, int offset, int lane){
    // definicion de offset y fragmentos
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;
    
    // (1) cargar datos de memoria global a A, B y C frags
    wmma::fill_fragment(a_frag, 1.0f);
    //wmma::fill_fragment(b_frag, 0.0f);
    wmma::fill_fragment(d_frag, 0.0f);

    // (2) mejora MMA multiples encadenados
    //const int bigoffset = gridDim.x * 32 * TCSQ;
    #pragma loop unroll
    for(int i=0; i<R; ++i){
        //if(threadIdx.x == 0) printf("offset %i \n",offset + TCSQ*32*(i+1));
        wmma::load_matrix_sync(b_frag, A + offset + 256*i, TCSIZE);
        wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);
    }

    // (3) preparando datos para segundo MMA
    wmma::fill_fragment(b_frag, 1.0f);
    #pragma loop unroll
    for(int i=0; i < 8; ++i){
        a_frag.x[i] = d_frag.x[i];
        a_frag.x[i+8] = d_frag.x[i];
    }
    wmma::fill_fragment(d_frag, 0.0f);

    // (4) MMA
    wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);

    // (5) Almacenar resultado
    if(lane == 0){
        //printf("block: %i, val %f\n",blockIdx.x,(float)d_frag.x[0]);
        return d_frag.x[0];
    }
    else return 0.0f;
}

__inline__ __device__ float block_reduce_tc(half *a, int offset){
	static __shared__ REAL shared[WARPSIZE];
	int tid = threadIdx.x;
	int lane = tid & (WARPSIZE-1);
	int wid = tid/WARPSIZE;
	REAL val = reduction_tc_warp(a, offset + wid*TCSQ*R, lane);
	if(lane == 0){
		shared[wid] = val;
    }
	__syncthreads();
	val = (tid < blockDim.x/WARPSIZE) ? shared[lane] : (REAL) 0.0f;
	if(wid == 0){
        val = shuffle_reduction(val);
    }
	return (float) val;
}
 __inline__ __device__ float block_reduce_shuffle(half val){
     static __shared__ half shared[WARPSIZE];
     int tid = threadIdx.x;
     int lane = tid & (WARPSIZE-1);
     int wid = tid/WARPSIZE;
     val = shuffle_reduction(val);
     if(lane == 0)
         shared[wid] = val;
 
     __syncthreads();
     val = (tid < blockDim.x/WARPSIZE) ? shared[lane] : (half) 0.f;
     if(wid == 0){
        val = shuffle_reduction(val);
     }
     return (float) val;
 }

__global__ void kernel_reduction_tc_blockshuffle(half *a, float *out, int N,int bs){
	//int offset = blockIdx.x * TCSQ * 32;       
	int offset = blockIdx.x * (bs * TCSQ * R); 
	if(offset < N){
		float sumf = block_reduce_tc(a, offset);
        if(threadIdx.x == 0){
            //printf("offset %i \n",offset);
            atomicAdd(out, sumf);
        }
	}
}
__global__ void kernel_reduction_tc_mixed(half *a, float *out, int bntc, int bns){
    int offset_shuffle = blockIdx.x * blockDim.x + threadIdx.x;
    float sum;
    if(offset_shuffle < bns*BSIZE){
        sum = block_reduce_shuffle(a[offset_shuffle]);
    }
    else{
        int offset_tc = (blockIdx.x-bns)*DIFF;
        sum = block_reduce_tc(a, offset_tc);
        //int offset_tc = (blockIdx.x - bns) * DIFF;
        //sum = block_reduce_tc(a, bns*BSIZE+offset_tc);
    }
     if(threadIdx.x == 0){
         atomicAdd(out, sum);
     }
}

__global__ void kernel_reduction_tc_theory(half* A, half* outd_m0, int n){
    grid_group grid = this_grid();
    //grid.sync();

    int wid = threadIdx.x/32;
    int offset = blockIdx.x*BSIZE + wid*32;
    // revisar aux     offset bloque  (BSIZE/WSIZE)*TCSQ*blockIdx.x    +   wid
    //int aux = wid+((BSIZE/TCSQ)*blockIdx.x*8);
    int aux = (BSIZE/WARPSIZE)*blockIdx.x + wid;
    //static __shared__ float num;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;

    wmma::fill_fragment(a_frag, 1.0f);
    wmma::fill_fragment(d_frag, 0.0f);

    wmma::load_matrix_sync(b_frag, A + offset, TCSIZE);
    wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);

    wmma::fill_fragment(b_frag, 1.0f);
         
    #pragma loop unroll
    for(int i=0; i < 8; ++i){
         a_frag.x[i] = d_frag.x[i];
         a_frag.x[i+8] = d_frag.x[i];
    }   
            
    wmma::fill_fragment(d_frag, 0.0f);
             
    // (4) MMA
    wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);
 
    // (5) Almacenar resultado
    #pragma loop unroll
    for(int i=0; i < BSIZE; i=i+WARPSIZE){
        if(threadIdx.x == i){
            outd_m0[aux] = d_frag.x[0];
        }
    }
    n = (n+255)/(256);
    grid.sync();

    while(n>=256){
        // (1) cargar datos de memoria global a A, B y C frags
        wmma::fill_fragment(a_frag, 1.0f);
        //wmma::fill_fragment(b_frag, 0.0f);
        wmma::fill_fragment(d_frag, 0.0f);
        
        // (2) mejora MMA multiples encadenados
        wmma::load_matrix_sync(b_frag, outd_m0 + offset, TCSIZE);
        wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);

        // (3) preparando datos para segundo MMA
        wmma::fill_fragment(b_frag, 1.0f);
        
        #pragma loop unroll
        for(int i=0; i < 8; ++i){
            a_frag.x[i] = d_frag.x[i];
            a_frag.x[i+8] = d_frag.x[i];
        }
        
        wmma::fill_fragment(d_frag, 0.0f);
        
        // (4) MMA
        wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);

        // (5) Almacenar resultado
        #pragma loop unroll
        for(int i=0; i < BSIZE; i=i+WARPSIZE){
            if(threadIdx.x == i){
                outd_m0[aux] = d_frag.x[0];
            }
        } 
        n = (n+255)/(256);
        grid.sync();
    }
    
    //siempre son menos de 256
    if(blockIdx.x == 0){
        int tid = threadIdx.x;
        half val;
        if(tid < n){
            val = block_reduce_shuffle(outd_m0[tid]);
        }
        if(threadIdx.x == 0){
            outd_m0[0] = val;   
        }
    }
 }
 
__global__ void kernel_reduction_shuffle(half *A, float *out, int n){
     int off = blockIdx.x * blockDim.x + threadIdx.x;
     if(off < n){
         REAL sum = A[off];
         sum = block_reduce_shuffle(sum);
         if(threadIdx.x == 0){
             atomicAdd(out, sum);
             //printf("sum %f \n",(float)sum);
         }
     }
 }
#endif
