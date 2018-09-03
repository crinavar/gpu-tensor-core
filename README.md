# gpu-tensor-core
A set of matmul operations for testing CUDA Tensor Core performance

# requirements
- A GPU with tensor cores (Volta architecture or later).
- CUDA 9.0+ 
- Linux

# compile
make

# run
./prog \<dev\> \<nmats\> \<alg\> \<prec\>
- \<dev\>: GPU id (0, 1, 2...)
- \<nmats\>: number of 16x16 matrices stored linearly and consecutively.
- \<alg\>:   
    - 0: standard
    - 1: standard + shared memory
    - 2: tensor-core with 16x2 blocks (one 16x16 matmul per block)
    - 3: tensor-core with 16x16 blocks (eight 16x16 matmuls per block)
    - 4: tensor-core with 16x16 blocks + shared memory
- \<mode\>: 
    - 0: FP16 input matrices
    - 1: FP32 input matrices (tensor-core algorithms cast to FP16 inside kernel)
