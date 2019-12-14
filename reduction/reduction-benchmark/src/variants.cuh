#pragma once

void warpshuffle_reduction(half *Adh, float *outd, long n, int REPEATS){
    dim3 block = dim3(BSIZE, 1, 1);
    dim3 grid = dim3((n + BSIZE -1)/BSIZE, 1, 1);
    for(int i=0; i<REPEATS; ++i){
        cudaMemset(outd, 0, sizeof(REAL)*1);
        kernel_reduction_shuffle<<<grid, block>>>(Adh, outd, n);  CUERR;
        cudaDeviceSynchronize();
    }
}

void recurrence_reduction(half *Adh, float *outd, half *outd_recA, half *outd_recB, long n, int REPEATS){
    dim3 block, grid;
    int rlimit = 1;
    half *temp, resh;
    float resf;
    int bs = BSIZE >> 5;
    for(int i=0; i<REPEATS; ++i){
        long dn = n;
        block = dim3(BSIZE, 1, 1);
        //grid = dim3((dn + TCSQ*bs-1)/(TCSQ*bs), 1, 1);
        grid = dim3((dn + (TCSQ*bs*(R)) - 1)/(TCSQ*bs*(R)), 1, 1);
        #ifdef DEBUG
            printf("executing recurrence for dn=%i >= rlimit =%i\n", dn, rlimit);
            printf("       grid (%i, %i, %i)    block(%i, %i, %i)\n\n\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
        #endif
        kernel_recurrence<<<grid, block>>>(Adh, outd_recA, dn);
        #ifdef DEBUG
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
            printf("DONE\n");
        #endif
        cudaDeviceSynchronize();
        while(dn > rlimit){
            dn = (dn + TCSQ-1)/TCSQ;
            //grid = dim3((dn + TCSQ*bs-1)/(TCSQ*bs), 1, 1);
            grid = dim3((dn + (TCSQ*bs*(R)) - 1)/(TCSQ*bs*(R)), 1, 1);
            #ifdef DEBUG
                printf("executing recurrence for dn=%i >= rlimit =%i\n", dn, rlimit);
                printf("       grid (%i, %i, %i)    block(%i, %i, %i)\n\n\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
            #endif
            kernel_recurrence<<<grid, block>>>(outd_recA, outd_recB, dn);
            cudaDeviceSynchronize();
            #ifdef DEBUG
                gpuErrchk( cudaPeekAtLastError() );
                gpuErrchk( cudaDeviceSynchronize() );
                printf("DONE\n");
            #endif
            temp = outd_recB;
            outd_recB = outd_recA;
            outd_recA = temp;
        }
    }
    cudaMemcpy(&resh, outd_recA, sizeof(half), cudaMemcpyDeviceToHost);    
    resf = __half2float(resh); 
    cudaMemcpy(outd, &resf, sizeof(float), cudaMemcpyHostToDevice);    
}

void chainedMMAs_reduction(half *Adh, float *outd, long n, int REPEATS){
    int bs = BSIZE >> 5;
    dim3 block = dim3(BSIZE, 1, 1);
    dim3 grid = dim3((n + (TCSQ*bs*(R)) - 1)/(TCSQ*bs*(R)), 1, 1);
    #ifdef DEBUG
        printf("grid (%i, %i, %i)    block(%i, %i, %i)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    #endif
    for(int i=0; i<REPEATS; ++i){
        cudaMemset(outd, 0, sizeof(REAL)*1);
        kernel_chainedMMAs<<<grid, block>>>(Adh, outd, n, bs);
        cudaDeviceSynchronize();
    }
}

void split_reduction(half *Adh, float *outd, long n, float factor_ns, int REPEATS){
    int bs = BSIZE >> 5;
    dim3 block = dim3(BSIZE, 1, 1);
    long nsh = (long)ceil(factor_ns*n);
    long ntc = n - nsh;
    int ns_blocks = (nsh + BSIZE-1)/BSIZE;
    int tc_blocks = (ntc + TCSQ*bs - 1)/(TCSQ*bs);
    dim3 grid = dim3(tc_blocks + ns_blocks, 1, 1);
    #ifdef DEBUG
        printf("ns_blocks %i, tc_blocks %i\n", ns_blocks, tc_blocks);
        printf("grid (%i, %i, %i)    block(%i, %i, %i)  DIFF %i\n", grid.x, grid.y, grid.z, block.x, block.y, block.z,DIFF);
    #endif
    for(int i=0; i<REPEATS; ++i){
        cudaMemset(outd, 0, sizeof(REAL)*1);
        kernel_split<<<grid, block>>>(n, Adh, outd, tc_blocks, ns_blocks);  CUERR;
        cudaDeviceSynchronize();
    }
}
