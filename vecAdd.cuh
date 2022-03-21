//
// Created by jundafeng on 3/22/22.
//

#ifndef TEST_CUDA_OPENCL_VECADD_CUH
#define TEST_CUDA_OPENCL_VECADD_CUH
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <CL/opencl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cxxtimer.hpp"

#define CPU "11th Gen Intel(R) Core(TM) i9-11980HK @ 2.60GHz"
#define INTEL_GPU "Intel(R) OpenCL HD Graphics "
#define NVIDIA_GPU "NVIDIA CUDA"
#define HYBRID_GPU "Intel(R) OpenCL HD Graphics + NVIDIA CUDA"

class vecAdd {
private:
   // sizes
   size_t N;
   size_t first_half_N;
   size_t second_half_N;
   size_t bytes, first_half_bytes, second_half_bytes;

   // Host input vectors
   float *h_a, *h_b, *h_c;

// buffers
   float* cuda_a, * cuda_b, * cuda_c;
   cl_mem cl_a, cl_b, cl_c;

// thread grid setting
   dim3 threadsPerBlock;
   dim3 numBlock;

   size_t localSize;
   size_t globalSize;

// API settings
   cudaStream_t cuda_stream;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel kernel;
   cl_int err;

public:
   vecAdd(unsigned int N, unsigned int factor, size_t intel_block_size , size_t cuda_block_size);
   ~vecAdd(){
      cudaFree(cuda_a);
      cudaFree(cuda_b);
      cudaFree(cuda_c);
      cudaStreamDestroy(cuda_stream);

      cudaFreeHost(h_a);
      cudaFreeHost(h_b);
      cudaFreeHost(h_c);

      clReleaseMemObject(cl_a);
      clReleaseMemObject(cl_b);
      clReleaseMemObject(cl_c);
      clReleaseProgram(program);
      clReleaseKernel(kernel);
      clReleaseCommandQueue(queue);
      clReleaseContext(context);
   };

   void init_host_data();
   void cpu_routine();
   void cuda_routine();
   void opencl_routine();

   void heterogeneous_overlap_preprocess();
   void heterogeneous_overlap_routine();

   void output_reset();
   void verify_result(const std::string&);
};


#endif //TEST_CUDA_OPENCL_VECADD_CUH
