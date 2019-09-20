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
    //wmma::fill_fragment(a_frag, 0.0f);
    // (3) preparando datos para segundo MMA
    //for(int i=0; i < 8; ++i){
    //    a_frag.x[i] = c_frag.x[i];
    //    a_frag.x[i+8] = c_frag.x[i];
    //}
    wmma::store_matrix_sync(As+off, c_frag, TCSIZE, wmma::mem_row_major);
    wmma::load_matrix_sync(a_frag, As+off, TCSIZE);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::fill_fragment(b_frag, 1.0f);
    wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
    
    wmma::store_matrix_sync(C + off, d_frag, TCSIZE, wmma::mem_row_major);
}

__global__ void mma_identity(half* A, half* B, REAL *C, int n){
    // offset for block
    int off = blockIdx.x * TCSQ;
    __shared__ half As[256*2];//As[TCSIZE*TCSIZE];
    // warp id
    int wid = threadIdx.x >> 5;
    // solo el primer warp trabajara, de ocho warps
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, REAL> d_frag;

    // MMA 1
    // (3) cargar datos de memoria global a A, B y C frags
    // cargar identidad en A, osea A = I
    //wmma::load_matrix_sync(a_frag, A + off, TCSIZE);
    wmma::load_matrix_sync(a_frag, A + off, TCSIZE);
    //wmma::fill_fragment(a_frag, 1.0f);
    // cargar datos de B
    wmma::load_matrix_sync(b_frag, B + off, TCSIZE);
    // inicializar C en cero
    wmma::fill_fragment(c_frag, 0.0f);
    // hacer multiplicacion C = I x B + [0] (es equivalente a copiar B en C)
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // MMA 2
    // ahora cargar identidad en B, osea B = I
    wmma::fill_fragment(a_frag, 0.0f);


    // copiar C en A a mano, A = C
        for(int i=0; i < 8; ++i){
            //if(threadIdx.x != 8){break;}
            //a_frag.x[i] = c_frag.x[i];
            int o = ((threadIdx.x >> 3) & 1) << 3;
            a_frag.x[i+o] = c_frag.x[i];
            a_frag.x[i] = c_frag.x[i];
            //a_frag.x[i+8] = 5.0f;c_frag.x[i];
        }

    // shared mem
    //wmma::store_matrix_sync(As+off, c_frag, TCSIZE, wmma::mem_row_major);
    //wmma::load_matrix_sync(a_frag, As+off, TCSIZE);


    // hacer C = 0
    wmma::fill_fragment(c_frag, 0.0f);
    // [DEBUG] FUERTEMENTE RELACIONADO AL BUG
    //wmma::load_matrix_sync(b_frag, A + off, TCSIZE);
    wmma::fill_fragment(b_frag, 1.0f);


    // hacer MMA D = A x I + [0], es decir D = A, y dado que A = C y C = B, => D = B
    wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
    // guardar D en C para imprimirla luego desde Host
    wmma::store_matrix_sync(C + off, d_frag, TCSIZE, wmma::mem_row_major);
}
#endif
