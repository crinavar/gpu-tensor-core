#ifndef _KERNEL_H
#define _KERNEL_H_
using namespace nvcuda;

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

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
        //val += __shfl_down_sync(0xFFFF, val, offset, WARPSIZE);
    return val;
}

// kernel
//__global__ void tc_reduction(half* A, int n){
__inline__ __device__ REAL tc_reduction(half *A, int offset, int lane){
    // definicion de offset y fragmentos
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;
    
    // (1) cargar datos de memoria global a A, B y C frags
    wmma::fill_fragment(a_frag, 1.0f);
    wmma::fill_fragment(b_frag, 0.0f);
    wmma::fill_fragment(d_frag, 0.0f);
    wmma::load_matrix_sync(b_frag, A + offset, TCSIZE);

    // (2) MMA
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
    //wmma::store_matrix_sync(A + off, d_frag, TCSIZE, wmma::mem_row_major);
    if(lane == 0){
        //A[blockIdx.x] = d_frag.x[0];
        //A[blockIdx.x] = 0.001f;
        //printf("tc_reduction return: %f \n",(float) d_frag.x[0]);
        return (REAL) d_frag.x[0];
        //return d_frag.x[0];
    }
    else return 0.0f;
}

__inline__ __device__ float block_reduce(half *a, int offset){
	static __shared__ REAL shared[WARPSIZE];               //se puede half?????
	int tid = threadIdx.x;
	int lane = tid & (WARPSIZE-1);
	int wid = tid/WARPSIZE;
	REAL val = tc_reduction(a, offset + wid*TCSQ, lane);
    //if(lane == 0){
    //    printf("soy thread %i y actuo desde a[%i], mi valor es %f\n", tid, offset+wid*TCSQ, val);
    //    printf("a[%i]=%f\n", offset+wid*TCSQ, (float)a[offset+wid*TCSQ]);
    //}
    //REAL val = shuffle_reduction(val);
	if(lane == 0){
		shared[wid] = val;
        //printf("tc_reduction val: %f \n",val);
    }
	__syncthreads();
	val = (tid < blockDim.x/WARPSIZE) ? shared[lane] : (REAL) 0.0f;
	if(wid == 0){
	    //shuffle_reduction
        val = shuffle_reduction(val);
	    //printf("shuffle val: %f \n",(float)val);
    }
	return (float) val;
}

__global__ void kernel_reduction(half *a, float *out, int N){
	//int tid = blockIdx.x * TCSQ;       
	int offset = blockIdx.x * (32*TCSQ);       
    //int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//if(tid < N){
	if(offset < N){
		//half sum = a[tid];
		float sumf = block_reduce(a, offset);
        //printf("sumf: %f \n",sumf);
        if(threadIdx.x == 0){
		    atomicAdd(out, sumf);
		    //printf("out: %f \n",out);
        }
	}
}
#endif
