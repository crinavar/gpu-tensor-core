#ifndef _KERNEL_H_
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



__global__ void matmuls_tc(half* A, half* B, REAL *C, int n){
    int off = blockIdx.x * TCSQ;
    __shared__ half As[256*2];//As[TCSIZE*TCSIZE];
    // solo el primer warp trabajara, de ocho warps
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, REAL> d_frag;

    // (3) cargar datos de memoria global a A, B y C frags
    //wmma::load_matrix_sync(a_frag, A + off, TCSIZE);
    wmma::load_matrix_sync(b_frag, B + off, TCSIZE);
    wmma::fill_fragment(a_frag, 1.0f);
    //wmma::fill_fragment(b_frag, (half)1.0f);
    wmma::fill_fragment(c_frag, 0.0f);

    // (4) operacion matrix-multiply-accumulate (MMA)
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::fill_fragment(a_frag, 0.0f);
    // (3) preparando datos para segundo MMA

    wmma::store_matrix_sync(As+off, c_frag, TCSIZE, wmma::mem_row_major);
    wmma::load_matrix_sync(a_frag, As+off, TCSIZE);
    
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::fill_fragment(b_frag, 1.0f);
    wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
    
    //if (threadIdx.x == 0) printf("---------> ADD: %f\n",(float)d_frag.x[0]);
    wmma::store_matrix_sync(C + off, d_frag, TCSIZE, wmma::mem_row_major);
}

#endif
