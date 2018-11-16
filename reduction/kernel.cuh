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

// kernel
__global__ void tc_reduction(half* A, int n){
    int off = blockIdx.x * TCSQ;

    // (1) el unico warp del bloque trabajara
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;
    
    // (2) cargar datos de memoria global a A, B y C frags
    wmma::load_matrix_sync(a_frag, A + off, TCSIZE);
    wmma::fill_fragment(b_frag, 1.0f);
    wmma::fill_fragment(c_frag, 0.0f);

    // (3) operacion matrix-multiply-accumulate (MMA)
    wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

    // (4) preparando datos para segundo MMA
    for(int i=0; i<d_frag.num_elements; ++i){
        a_frag.x[i] = d_frag.x[i];
    }
    wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

    // (5) guardar datos de vuelta en memorial global
    //A[blockIdx.x] = d_frag.x[0];
    //printf("thread (%i, %i, %i)    valor es %f\n", threadIdx.x, threadIdx.y, threadIdx.z, (float)A[blockIdx.x]);
    wmma::store_matrix_sync(A + off, d_frag, TCSIZE, wmma::mem_row_major);
}
#endif
