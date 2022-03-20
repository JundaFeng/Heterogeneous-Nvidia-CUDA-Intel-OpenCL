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
#include <cuda_runtime_api.h>
#include "cxxtimer.hpp"

// Host input vectors
float *h_a, *h_b, *h_c;
const unsigned int N = 160000000;
const unsigned int half_N = N/16;
size_t bytes, first_half_bytes, second_half_bytes;


// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                       "\n" \
// "#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n"
                                                                 "__kernel void vecAdd(  __global float *a,                       \n" \
"                       __global float *b,                       \n" \
"                       __global float *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n){                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"    }                                                           \n" \
"}                                                               \n" \
                                                                "\n" ;


// GPU CUDA Kernel
__global__ void vec_add_kernel(const float* in1, const float* in2, float* out, int len){
   int idx = blockDim.x*blockIdx.x+threadIdx.x;
   if(idx<len){
      out[idx] = in1[idx]+in2[idx];
   }
}

// CPU for verification
void cpu_routine(){
   for(int idx=0;idx<N;idx++){
      h_c[idx] = h_a[idx]+h_b[idx];
   }
   float sum = 0;
   for (int i = 0; i < N; i++)
      sum += h_c[i];
   std::cout << "Result on CPU : " << sum << std::endl;
}

int opencl_routine(){
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel kernel;
   cl_int err;

   // Bind to platform
   cl_uint num_pltfs, num_devs, num_entries = 3;
   std::vector<cl_platform_id> platforms(num_entries);
   std::vector<cl_device_id> device_ids(num_entries);
   err = clGetPlatformIDs(num_entries, &platforms[0], &num_pltfs);
   if (err != CL_SUCCESS) {
      std::cout << "Cannot get platform" << std::endl;
      return -1;
   }

   err = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, num_entries, &device_ids[0], &num_devs);
   if (err != CL_SUCCESS) {
      switch (err) {
         case CL_INVALID_PLATFORM: std::cout << "CL_INVALID_PLATFORM" << std::endl;
         case CL_INVALID_DEVICE_TYPE: std::cout << "CL_INVALID_DEVICE_TYPE" << std::endl;
         case CL_INVALID_VALUE: std::cout << "CL_INVALID_VALUE" << std::endl;
         case CL_DEVICE_NOT_FOUND: std::cout << "CL_DEVICE_NOT_FOUND" << std::endl;
         default: std::cout << "Cannot get device - Other reason" << std::endl;
      }
      return -1;
   }

   cl_device_id device_id = device_ids[0];

   context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
   if (err != CL_SUCCESS) {
      std::cout << "Create context failed" << std::endl;
      return -1;
   }

   queue = clCreateCommandQueue(context, device_id, 0, &err);
   if (err != CL_SUCCESS) {
      std::cout << "Create command queue failed" << std::endl;
      return -1;
   }


   cl_mem d_a, d_b, d_c;
   d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_a, nullptr);
   d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_b, nullptr);
   d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
   if (d_a == nullptr || d_b == nullptr || d_c == nullptr) {
      std::cout << "Create buffer failed" << std::endl;
      return -1;
   }

   size_t globalSize, localSize = 16;
   globalSize = static_cast<size_t>(ceil(N / (float) localSize) * localSize);

   // Write our data set into the input array in device memory
   err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes, h_a, 0, nullptr, nullptr);
   err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes, h_b, 0, nullptr, nullptr);
   if (err != CL_SUCCESS) {
      std::cout << "Enqueue Write Buffer failed" << std::endl;
      return -1;
   }

   program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, nullptr, &err);
   if (program == nullptr) {
      std::cout << "Create program failed" << std::endl;
      return -1;
   }

   clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
   if (err != CL_SUCCESS) {
      std::cout << "Build program failed" << std::endl;
      return -1;
   }

   kernel = clCreateKernel(program, "vecAdd", &err);
   if (kernel == nullptr) {
      std::cout << "Create kernel failed" << std::endl;
      return -1;
   }


   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
   err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &N);
   if (err != CL_SUCCESS) {
      std::cout << "Set kernel arg failed" << std::endl;
      return -1;
   }
   timer_start("opencl_routine_kernel_time",'m');
   err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                0, nullptr, nullptr);
   if (err != CL_SUCCESS) {
      std::cout << "Run kernel failed" << std::endl;
      return -1;
   }

   clFinish(queue);
   timer_stop('m');

   clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, nullptr, nullptr);
   if (err != CL_SUCCESS) {
      std::cout << "Read data failed" << std::endl;
      return -1;
   }

   float sum = 0;
   for (int i = 0; i < N; i++)
      sum += h_c[i];
   std::cout << "Result on Intel(R) OpenCL HD Graphics : " << sum << std::endl;
   clReleaseMemObject(d_a);
   clReleaseMemObject(d_b);
   clReleaseMemObject(d_c);
   clReleaseProgram(program);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseContext(context);
}

#define CH_CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } \
}

#define CUDA_SAFE_CALL(call) CH_CUDA_SAFE_CALL(call)

int cuda_routine(){
   float* d_a, * d_b, * d_c;

   CUDA_SAFE_CALL(cudaMalloc((void**) &d_a, bytes));
   CUDA_SAFE_CALL(cudaMalloc((void**) &d_b, bytes));
   CUDA_SAFE_CALL(cudaMalloc((void**) &d_c, bytes));

   CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, bytes,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemset(d_c, 0, bytes));

   dim3 threadsPerBlock(512);
   dim3 numBlock((N-1)/threadsPerBlock.x+1);
   timer_start("cuda_routine_kernel_time",'m');
   vec_add_kernel<<<numBlock, threadsPerBlock>>>(d_a, d_b, d_c, N);
   CUDA_SAFE_CALL(cudaDeviceSynchronize());
   timer_stop('m');
   CUDA_SAFE_CALL(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

   float sum = 0;
   for (int i = 0; i < N; i++)
      sum += h_c[i];
   std::cout << "Result on NVIDIA CUDA : " << sum << std::endl;

   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);
}

int heterogeneous_routine(){
   float* cuda_a, * cuda_b, * cuda_c;
   dim3 numBlock, threadsPerBlock;

   cl_mem cl_a, cl_b, cl_c;
   cl_context context;
   cl_command_queue queue;
   cl_program program;
   cl_kernel kernel;
   cl_int err;
   cl_uint num_pltfs, num_devs, num_entries = 3;
   size_t globalSize, localSize;
   {
      std::vector<cl_platform_id> platforms(num_entries);
      std::vector<cl_device_id> device_ids(num_entries);
      err = clGetPlatformIDs(num_entries, &platforms[0], &num_pltfs);
      if (err != CL_SUCCESS) {
         std::cout << "Cannot get platform" << std::endl;
         return -1;
      }

      err = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, num_entries, &device_ids[0], &num_devs);
      if (err != CL_SUCCESS) {
         switch (err) {
            case CL_INVALID_PLATFORM: std::cout << "CL_INVALID_PLATFORM" << std::endl;
            case CL_INVALID_DEVICE_TYPE: std::cout << "CL_INVALID_DEVICE_TYPE" << std::endl;
            case CL_INVALID_VALUE: std::cout << "CL_INVALID_VALUE" << std::endl;
            case CL_DEVICE_NOT_FOUND: std::cout << "CL_DEVICE_NOT_FOUND" << std::endl;
            default: std::cout << "Cannot get device - Other reason" << std::endl;
         }
         return -1;
      }

      cl_device_id device_id = device_ids[0];

      context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
      if (err != CL_SUCCESS) {
         std::cout << "Create context failed" << std::endl;
         return -1;
      }

      queue = clCreateCommandQueue(context, device_id, 0, &err);
      if (err != CL_SUCCESS) {
         std::cout << "Create command queue failed" << std::endl;
         return -1;
      }

      cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, first_half_bytes, h_a, nullptr);
      cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, first_half_bytes, h_b, nullptr);
      cl_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, first_half_bytes, nullptr, nullptr);
      if (cl_a == nullptr || cl_b == nullptr || cl_c == nullptr) {
         std::cout << "Create buffer failed" << std::endl;
         return -1;
      }

      localSize = 16;
      globalSize = static_cast<size_t>(ceil(half_N / (float) localSize) * localSize);

      // Write our data set into the input array in device memory
      err = clEnqueueWriteBuffer(queue, cl_a, CL_TRUE, 0, first_half_bytes, h_a, 0, nullptr, nullptr);
      err |= clEnqueueWriteBuffer(queue, cl_b, CL_TRUE, 0, first_half_bytes, h_b, 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
         std::cout << "Enqueue Write Buffer failed" << std::endl;
         return -1;
      }

      program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, nullptr, &err);
      if (program == nullptr) {
         std::cout << "Create program failed" << std::endl;
         return -1;
      }

      clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
      if (err != CL_SUCCESS) {
         std::cout << "Build program failed" << std::endl;
         return -1;
      }

      kernel = clCreateKernel(program, "vecAdd", &err);
      if (kernel == nullptr) {
         std::cout << "Create kernel failed" << std::endl;
         return -1;
      }


      err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_a);
      err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_b);
      err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_c);
      err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &half_N);
      if (err != CL_SUCCESS) {
         std::cout << "Set kernel arg failed" << std::endl;
         return -1;
      }
   } // OpenCL Preprocess

   {
      CUDA_SAFE_CALL(cudaMalloc((void**) &cuda_a, second_half_bytes));
      CUDA_SAFE_CALL(cudaMalloc((void**) &cuda_b, second_half_bytes));
      CUDA_SAFE_CALL(cudaMalloc((void**) &cuda_c, second_half_bytes));

      CUDA_SAFE_CALL(cudaMemcpy(cuda_a, h_a+half_N, second_half_bytes, cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(cuda_b, h_b+half_N, second_half_bytes, cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemset(cuda_c, 0, second_half_bytes));

      threadsPerBlock = 512;
      numBlock = (N-half_N-1)/threadsPerBlock.x+1;
   } // CUDA Preprocess

   timer_start("heterogeneous_routine_kernel_time",'m');
   // Launch OpenCL Kernel
   err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                0, nullptr, nullptr);
   // Launch CUDA Kernel
   vec_add_kernel<<<numBlock, threadsPerBlock>>>(cuda_a, cuda_b, cuda_c, N);


   if (err != CL_SUCCESS) {
      std::cout << "Run kernel failed" << std::endl;
      return -1;
   }
   clFinish(queue);
   CUDA_SAFE_CALL(cudaDeviceSynchronize());
   timer_stop('m');


   {
      CUDA_SAFE_CALL(cudaMemcpy(h_c+half_N, cuda_c, second_half_bytes, cudaMemcpyDeviceToHost));
      cudaFree(cuda_a);
      cudaFree(cuda_b);
      cudaFree(cuda_c);
   } // CUDA Postprocess

   {
      err = clEnqueueReadBuffer(queue, cl_c, CL_FALSE, 0, first_half_bytes, h_c, 0, nullptr, nullptr);

      if (err != CL_SUCCESS) {
         std::cout << "Read data failed" << std::endl;
         return -1;
      }
      clReleaseMemObject(cl_a);
      clReleaseMemObject(cl_b);
      clReleaseMemObject(cl_c);
      clReleaseProgram(program);
      clReleaseKernel(kernel);
      clReleaseCommandQueue(queue);
      clReleaseContext(context);
   } // OpenCL Postprocess

   float sum = 0;
   for (int i = 0; i < N; i++)
      sum += h_c[i];
   std::cout << "Result on Intel(R) OpenCL HD Graphics + NVIDIA CUDA: " << sum << std::endl;
}

int main(int argc, char* argv[]) {
   bytes = N * sizeof(float);
   first_half_bytes=  half_N * sizeof(float);
   second_half_bytes = bytes - first_half_bytes;

   std::cout << "Number of bytes in Giga: " << 3*static_cast<float>(bytes)/pow(10,9) << std::endl;

   // Allocate memory for each vector on host
   h_a = (float *) malloc(bytes);
   h_b = (float *) malloc(bytes);
   h_c = (float *) malloc(bytes);

   for (int i = 0; i < N; i++) {
      h_a[i] = 1.0*i/N;
      h_b[i] = 1.0*i/N;
   }

   timer_start("cpu_routine",'m');
   cpu_routine();   memset(h_c, 0, bytes);
   timer_stop('m');

   timer_start("opencl_routine",'m');
   opencl_routine();   memset(h_c, 0, bytes);
   timer_stop('m');

   timer_start("cuda_routine",'m');
   cuda_routine();   memset(h_c, 0, bytes);
   timer_stop('m');

   timer_start("heterogeneous_routine",'m');
   heterogeneous_routine();
   timer_stop('m');

   return 0;
}
