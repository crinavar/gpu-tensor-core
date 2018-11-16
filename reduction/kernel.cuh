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

// kernel
__global__ void warpReduceSumTC(half* A, REAL* C, int n){
    int off = blockIdx.x * TCSQ;

    // solo el primer warp trabajara, de ocho warps
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, REAL> c_frag;
    

    // (3) cargar datos de memoria global a A, B y C frags
    wmma::load_matrix_sync(a_frag, A + off, TCSIZE);
    wmma::fill_fragment(b_frag, 1.0f);
    wmma::fill_fragment(c_frag, 0.0f);

    // (4) operacion matrix-multiply-accumulate (MMA)
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // (5) guardar datos de vuelta en memorial global
    wmma::store_matrix_sync(C + off, c_frag, TCSIZE, wmma::mem_row_major);
}
#endif
