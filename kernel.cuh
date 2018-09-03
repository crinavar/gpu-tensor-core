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

// kernel
__global__ void matmuls_basic(REAL* A, REAL* B, REAL *C, int n){
    int off = blockIdx.x * (TCSIZE*TCSIZE);
    int tid = off + (threadIdx.y*TCSIZE + threadIdx.x);
    for(int i=0; i<TCSIZE; ++i){
        C[tid] += A[off + threadIdx.y*TCSIZE + i]*B[off + i*TCSIZE + threadIdx.x];
    }
}

__global__ void matmuls_sm(REAL* A, REAL* B, REAL *C, int n){
    __shared__ REAL As[TCSIZE*TCSIZE];
    __shared__ REAL Bs[TCSIZE*TCSIZE];
    __shared__ REAL Cs[TCSIZE*TCSIZE];
    int off = blockIdx.x * (TCSIZE*TCSIZE);
    int ltid = threadIdx.y*TCSIZE + threadIdx.x;
    // (1) cargar datos de global mem a cache
    As[ltid] = A[off + ltid];
    Bs[ltid] = B[off + ltid];
    Cs[ltid] = 0.0f;
    // (2) sync
    __syncthreads();
    // (3) hacer multiplicacion usando As, Bs, Cs 
    #pragma unroll
    for(int i=0; i<TCSIZE; ++i){
        Cs[ltid] += As[threadIdx.y*TCSIZE + i]*Bs[i*TCSIZE + threadIdx.x];
    }
    // (4) copiar Cs a memoria global.
    C[off + ltid] = Cs[ltid];
}






__global__ void matmuls_tc(half* A, half* B, REAL *C, int n){
    int off = blockIdx.x * TCSQ;

    // solo el primer warp trabajara, de ocho warps
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, REAL> c_frag;

    // (3) cargar datos de memoria global a A, B y C frags
    wmma::load_matrix_sync(a_frag, A + off, TCSIZE);
    wmma::load_matrix_sync(b_frag, B + off, TCSIZE);
    //wmma::fill_fragment(a_frag, (half)1.0f);
    //wmma::fill_fragment(b_frag, (half)1.0f);
    wmma::fill_fragment(c_frag, 0.0f);

    // (4) operacion matrix-multiply-accumulate (MMA)
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // (5) guardar datos de vuelta en memorial global
    wmma::store_matrix_sync(C + off, c_frag, TCSIZE, wmma::mem_row_major);

}

__global__ void matmuls_tc_block(half* A, half* B, REAL *C, int n){
    int off = blockIdx.x * TCSQ * 8;
    int ltid = threadIdx.y*TCSIZE + threadIdx.x;
    int woff = (ltid >> 5) << 8;
    /*
    __shared__ half As[8*TCSQ];
    __shared__ half Bs[8*TCSQ];
    int ltid = threadIdx.y*TCSIZE + threadIdx.x;
    //int woff = (ltid >> 5)*TCSQ;
    int woff = (ltid >> 5) << 8;
    // (1) cargar datos de global mem a cache
    #pragma unroll
    for(int i=0; i<8; ++i){
        As[TCSQ*i + ltid] = A[off + i*TCSQ + ltid];
        Bs[TCSQ*i + ltid] = B[off + i*TCSQ + ltid];
    }
    // (2) sync
    __syncthreads();
    */

    // solo el primer warp trabajara, de ocho warps
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, REAL> c_frag;

    // (3) cargar datos de memoria global a A, B y C frags
    //wmma::load_matrix_sync(a_frag, As + woff, TCSIZE);
    //wmma::load_matrix_sync(b_frag, Bs + woff, TCSIZE);
    wmma::load_matrix_sync(a_frag, A + off + woff, TCSIZE);
    wmma::load_matrix_sync(b_frag, B + off + woff, TCSIZE);
    wmma::fill_fragment(c_frag, 0.0f);

    // (4) operacion matrix-multiply-accumulate (MMA)
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // (5) guardar datos de vuelta en memorial global
    wmma::store_matrix_sync(C + off + woff, c_frag, TCSIZE, wmma::mem_row_major);
}

__global__ void matmuls_tc_block_sm(half* A, half* B, REAL *C, int n){
    int off = blockIdx.x * TCSQ * 8;
    __shared__ half As[8*TCSQ];
    __shared__ half Bs[8*TCSQ];
    int ltid = threadIdx.y*TCSIZE + threadIdx.x;
    //int woff = (ltid >> 5)*TCSQ;
    int woff = (ltid >> 5) << 8;
    // (1) cargar datos de global mem a cache
    #pragma unroll
    for(int i=0; i<8; ++i){
        As[TCSQ*i + ltid] = A[off + i*TCSQ + ltid];
        Bs[TCSQ*i + ltid] = B[off + i*TCSQ + ltid];
    }
    // (2) sync
    __syncthreads();

    // solo el primer warp trabajara, de ocho warps
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, REAL> c_frag;

    // (3) cargar datos de memoria global a A, B y C frags
    wmma::load_matrix_sync(a_frag, As + woff, TCSIZE);
    wmma::load_matrix_sync(b_frag, Bs + woff, TCSIZE);
    wmma::fill_fragment(c_frag, 0.0f);

    // (4) operacion matrix-multiply-accumulate (MMA)
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // (5) guardar datos de vuelta en memorial global
    wmma::store_matrix_sync(C + off + woff, c_frag, TCSIZE, wmma::mem_row_major);
}
#endif
