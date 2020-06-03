/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Simple example of DeviceReduce::Sum().
 *
 * Sums an array of int keys.
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_device_reduce.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

#include "nvmlPower.hpp"
#include "../../test/test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory
//int g_timing_iterations = 30;


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Initialize problem
 */



double Initialize_normal(float *h_in, int num_items, int seed)
{
     //srand(412);
     double inclusive = 0.0;
     //std::random_device rd;
     std::mt19937 gen{seed};
     std::normal_distribution<> d{0,1};
     for (int i = 0; i < num_items; ++i)
     {
         //h_in[i]= (float)rand()/((float)(RAND_MAX)*1000);
         h_in[i] = d(gen);
         inclusive += h_in[i];

     }
    /* std::map<int, int> hist{};
     for(int n=0; n<10000; ++n) {
         ++hist[std::round(d(gen))];
     }
     for(auto p : hist) {
         std::cout << std::setw(2)
                   << p.first << ' ' << std::string(p.second/200, '*') << '\n';
     }*/
     return inclusive;
}

double Initialize_uniform(float *h_in, int num_items, int seed)
 {
    //srand(12);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> d(0, 1);
    double inclusive = 0.0;
    for (int i = 0; i < num_items; ++i)
    {
        h_in[i] = (float) d(gen);
        //h_in[i] = (float)rand()/((float)(RAND_MAX)*1000);
        inclusive += h_in[i];
 
    }
    return inclusive;
 }


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv){
    if(argc != 6){
        fprintf(stderr, "run as ./cub-float dev n dist seed repeat\n\n"); fflush(stdout);
        exit(EXIT_FAILURE);
    }
    srand(time(NULL));
    //problem size by console
    int num_items = atoi(argv[2]);
    float h_out=0.0;

    int dev = atoi(argv[1]);
    cudaSetDevice(dev);

    int dist = atoi(argv[3]);

    int seed = atoi(argv[4]);

    int repeat = atoi(argv[5]);
    int g_timing_iterations = repeat;
    // Initialize command line
    printf("num items = %i\n", num_items);
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);

    /*
    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--device=<device-id>] "
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    //printf("cub::DeviceReduce::Sum() %d items (%d-byte elements)\n",
        num_items, (int) sizeof(int));
    fflush(stdout);
*/
    // Allocate host arrays
    float* h_in = new float[num_items];


    // Initialize problem and solution
    double  h_reference;
    if (dist==0){
        h_reference = Initialize_normal(h_in, num_items,seed);
    }
    else{
        h_reference = Initialize_uniform(h_in, num_items, seed);
    }
        //printf("\tSuma calculada en Host:%f\n",h_reference);
 

    // Allocate problem device arrays
    float *d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in, sizeof(float) * num_items));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice));

    // Allocate device output array
    float *d_out = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out, sizeof(float) * 1));

    // Request and allocate temporary storage
    void            *d_temp_storage = NULL;
    size_t          temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));



    // Run
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));



    // Check for correctness (and display results, if specified)
    int compare = CompareDeviceResults(&h_reference, d_out, 1, g_verbose, g_verbose);
    //printf("\t%s", compare ? "FAIL" : "PASS");
    //AssertEquals(0, compare);

     cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    //printf("\tsuma calculada en device: %f\n\n",h_out);


        // Run this several times and average the performance results
    GpuTimer    timer;
    float       elapsed_millis          = 0.0;

 //   printf("\tnumero de pruebas para tiempo promedio: %d\n", g_timing_iterations);
    
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice));
    timer.Start();

    #ifdef POWER
        GPUPowerBegin("CUB-float");
	#ifdef POWER_DEBUG
		printf("Started Measuring power, press enter...\n"); fflush(stdout);
		getchar();
	#endif
    #endif
    for (int i = 0; i < g_timing_iterations; ++i)
    {
        // Copy problem to device
        //CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * num_items, cudaMemcpyHostToDevice));

        // Run aggregate
        CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items));

        cudaDeviceSynchronize();
        //elapsed_millis += timer.ElapsedMillis();
    }
    #ifdef POWER
	#ifdef POWER_DEBUG
		printf("DONE: press enter to stop\n");
		getchar();
	#endif
        GPUPowerEnd();
    #endif
    timer.Stop();
    elapsed_millis = timer.ElapsedMillis();


    // Check for kernel errors and STDIO from the kernel, if any
    CubDebugExit(cudaPeekAtLastError());
    CubDebugExit(cudaDeviceSynchronize());

    // Display timing results
    float avg_millis            = elapsed_millis / g_timing_iterations;

    //printf("\ttiempo promedio (mili segundos): %.4f\n", avg_millis);
    printf("%f,%f,%f,%f,%f\n",avg_millis,(float)h_out,h_reference,fabs((float)h_out - h_reference),fabs(100.0f*fabs((float)h_out - h_reference)/h_reference));
    // Cleanup
    if (h_in) delete[] h_in;
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_out) CubDebugExit(g_allocator.DeviceFree(d_out));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));

    //printf("\n\n");

    return 0;
}



