//
// Created by jundafeng on 3/22/22.
//

#include <unistd.h>
#include "vecAdd.cuh"

std::string getErrorString(cl_int error)
{
   switch(error){
      // run-time and JIT compiler errors
      case 0: return "CL_SUCCESS";
      case -1: return "CL_DEVICE_NOT_FOUND";
      case -2: return "CL_DEVICE_NOT_AVAILABLE";
      case -3: return "CL_COMPILER_NOT_AVAILABLE";
      case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
      case -5: return "CL_OUT_OF_RESOURCES";
      case -6: return "CL_OUT_OF_HOST_MEMORY";
      case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
      case -8: return "CL_MEM_COPY_OVERLAP";
      case -9: return "CL_IMAGE_FORMAT_MISMATCH";
      case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
      case -11: return "CL_BUILD_PROGRAM_FAILURE";
      case -12: return "CL_MAP_FAILURE";
      case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
      case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
      case -15: return "CL_COMPILE_PROGRAM_FAILURE";
      case -16: return "CL_LINKER_NOT_AVAILABLE";
      case -17: return "CL_LINK_PROGRAM_FAILURE";
      case -18: return "CL_DEVICE_PARTITION_FAILED";
      case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

         // compile-time errors
      case -30: return "CL_INVALID_VALUE";
      case -31: return "CL_INVALID_DEVICE_TYPE";
      case -32: return "CL_INVALID_PLATFORM";
      case -33: return "CL_INVALID_DEVICE";
      case -34: return "CL_INVALID_CONTEXT";
      case -35: return "CL_INVALID_QUEUE_PROPERTIES";
      case -36: return "CL_INVALID_COMMAND_QUEUE";
      case -37: return "CL_INVALID_HOST_PTR";
      case -38: return "CL_INVALID_MEM_OBJECT";
      case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
      case -40: return "CL_INVALID_IMAGE_SIZE";
      case -41: return "CL_INVALID_SAMPLER";
      case -42: return "CL_INVALID_BINARY";
      case -43: return "CL_INVALID_BUILD_OPTIONS";
      case -44: return "CL_INVALID_PROGRAM";
      case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
      case -46: return "CL_INVALID_KERNEL_NAME";
      case -47: return "CL_INVALID_KERNEL_DEFINITION";
      case -48: return "CL_INVALID_KERNEL";
      case -49: return "CL_INVALID_ARG_INDEX";
      case -50: return "CL_INVALID_ARG_VALUE";
      case -51: return "CL_INVALID_ARG_SIZE";
      case -52: return "CL_INVALID_KERNEL_ARGS";
      case -53: return "CL_INVALID_WORK_DIMENSION";
      case -54: return "CL_INVALID_WORK_GROUP_SIZE";
      case -55: return "CL_INVALID_WORK_ITEM_SIZE";
      case -56: return "CL_INVALID_GLOBAL_OFFSET";
      case -57: return "CL_INVALID_EVENT_WAIT_LIST";
      case -58: return "CL_INVALID_EVENT";
      case -59: return "CL_INVALID_OPERATION";
      case -60: return "CL_INVALID_GL_OBJECT";
      case -61: return "CL_INVALID_BUFFER_SIZE";
      case -62: return "CL_INVALID_MIP_LEVEL";
      case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
      case -64: return "CL_INVALID_PROPERTY";
      case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
      case -66: return "CL_INVALID_COMPILER_OPTIONS";
      case -67: return "CL_INVALID_LINKER_OPTIONS";
      case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

         // extension errors
      case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
      case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
      case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
      case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
      case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
      case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
      default: return "Unknown OpenCL error";
   }
}


// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                       "\n" \
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
__global__ void vec_add_kernel(const float* in1, const float* in2, float* out, unsigned int len){
   unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;
   if(idx<len){
      out[idx] = in1[idx]+in2[idx];
   }
}

// CPU for verification
void vecAdd::cpu_routine(){
   for(int idx=0;idx<N;idx++){
      h_c[idx] = h_a[idx]+h_b[idx];
   }
}

vecAdd::vecAdd(unsigned int vec_size, unsigned int divide_factor, size_t intel_block_size, size_t cuda_block_size)
{
   N = vec_size;
   first_half_N = N/divide_factor;
   second_half_N = N - first_half_N;

   localSize  = intel_block_size;
   threadsPerBlock = cuda_block_size;

   // data divide and conquer
   bytes = N * sizeof(float);
   first_half_bytes=  first_half_N * sizeof(float);
   second_half_bytes = bytes - first_half_bytes;

   // OpenCL device settings
   // CUDA Device settings
   cudaStreamCreate(&cuda_stream);

   std::vector<cl_platform_id> platforms(3); // OpenCL only handle intel CPU and GPU
   err = clGetPlatformIDs(3, &platforms[0], nullptr);
   if (err != CL_SUCCESS) {
      std::cerr << "Cannot get platform" << std::endl;
      exit(-1);
   }

   cl_device_id device_id;
   err = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
   if (err != CL_SUCCESS) {
      switch (err) {
         case CL_INVALID_PLATFORM: std::cerr << "CL_INVALID_PLATFORM" << std::endl;
         case CL_INVALID_DEVICE_TYPE: std::cerr << "CL_INVALID_DEVICE_TYPE" << std::endl;
         case CL_INVALID_VALUE: std::cerr << "CL_INVALID_VALUE" << std::endl;
         case CL_DEVICE_NOT_FOUND: std::cerr << "CL_DEVICE_NOT_FOUND" << std::endl;
         default: std::cerr << "Cannot get device - Other reason" << std::endl;
      }
      exit(-1);
   }

   context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
   if (err != CL_SUCCESS) {
      std::cerr << "Create context failed" << std::endl;
      exit(-1);
   }

   queue = clCreateCommandQueue(context, device_id, 0, &err);
   if (err != CL_SUCCESS) {
      std::cerr << "Create command queue failed" << std::endl;
      exit(-1);
   }

   program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, nullptr, &err);
   if (program == nullptr) {
      std::cerr << "Create program failed" << std::endl;
      exit(-1);
   }

   clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
   if (err != CL_SUCCESS) {
      std::cerr << "Build program failed" << std::endl;
      exit(-1);
   }

   kernel = clCreateKernel(program, "vecAdd", &err);
   if (kernel == nullptr) {
      std::cerr << "Create kernel failed" << std::endl;
      exit(-1);
   }

   std::cout << "Number of bytes in Giga: " << 3*static_cast<float>(bytes)/pow(10,9) << std::endl;

   // Allocate memory for each vector on host
   cudaMallocHost((void**)&h_a, bytes);
   cudaMallocHost((void**)&h_b, bytes);
   cudaMallocHost((void**)&h_c, bytes);
}

void vecAdd::init_host_data() {
   for (int i = 0; i < N; i++) {
      h_a[i] = 1.0*i/N;
      h_b[i] = 1.0*i/N;
   }
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

void vecAdd::cuda_routine(){
   numBlock = (N-1)/threadsPerBlock.x+1;

   timer_start("cuda_routine_preprocess_time",'u');
   CUDA_SAFE_CALL(cudaMalloc((void**) &cuda_a, bytes));
   CUDA_SAFE_CALL(cudaMalloc((void**) &cuda_b, bytes));
   CUDA_SAFE_CALL(cudaMalloc((void**) &cuda_c, bytes));
   timer_stop('u');

   timer_start("cuda_routine_memcpy_h2d_time",'u');
   CUDA_SAFE_CALL(cudaMemcpy(cuda_a, h_a, bytes,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(cuda_b, h_b, bytes, cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemset(cuda_c, 0, bytes));
   timer_stop('u');

   timer_start("cuda_routine_kernel_time",'u');
   vec_add_kernel<<<numBlock, threadsPerBlock>>>(cuda_a, cuda_b, cuda_c, N);
   CUDA_SAFE_CALL(cudaDeviceSynchronize());
   timer_stop('u');

   timer_start("cuda_routine_memcpy_d2h_time",'u');
   CUDA_SAFE_CALL(cudaMemcpy(h_c, cuda_c, bytes, cudaMemcpyDeviceToHost));
   timer_stop('u');

   timer_start("cuda_routine_postprocess_time",'u');
   cudaFree(cuda_a);
   cudaFree(cuda_b);
   cudaFree(cuda_c);
   timer_stop('u');
}


void vecAdd::opencl_routine(){
   globalSize = static_cast<size_t>(((N-1) / localSize+1) * localSize);
   cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
   cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
   cl_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
   if (cl_a == nullptr || cl_b == nullptr || cl_c == nullptr) {
      std::cerr << "Create buffer failed" << std::endl;
      exit(-1);
   }

   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_a);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_b);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_c);
   err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &N);
   if (err != CL_SUCCESS) {
      std::cerr << "Set kernel arg failed: "+getErrorString(err) << std::endl;
      exit(-1);
   }

   timer_start("opencl_routine_memcpy_h2d_time",'u');
   // Write our data set into the input array in device memory
   err = clEnqueueWriteBuffer(queue, cl_a, CL_TRUE, 0, bytes, h_a, 0, nullptr, nullptr);
   err |= clEnqueueWriteBuffer(queue, cl_b, CL_TRUE, 0, bytes, h_b, 0, nullptr, nullptr);
   if (err != CL_SUCCESS) {
      std::cerr << "Enqueue Write Buffer failed" << std::endl;
      exit(-1);
   }
   timer_stop('u');

   timer_start("opencl_routine_kernel_time",'u');
   err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                0, nullptr, nullptr);
   if (err != CL_SUCCESS) {
      std::cerr << "Run kernel failed" << std::endl;
      exit(-1);
   }

   clFinish(queue);
   timer_stop('u');

   timer_start("opencl_routine_memcpy_d2h_time",'u');
   err = clEnqueueReadBuffer(queue, cl_c, CL_TRUE, 0, bytes, h_c, 0, nullptr, nullptr);
   if (err != CL_SUCCESS) {
      std::cerr << "Read data failed" << std::endl;
      exit(-1);
   }
   clFinish(queue);
   timer_stop('u');
//
//   timer_start("opencl_routine_postprocess_time",'u');
//   clReleaseMemObject(cl_a);
//   clReleaseMemObject(cl_b);
//   clReleaseMemObject(cl_c);
//
//   timer_stop('u');
}



void vecAdd::heterogeneous_overlap_preprocess(){
   timer_start("heterogeneous_overlap_routine_preprocess_time",'u');
   globalSize = static_cast<size_t>(((first_half_N-1) / localSize+1) * localSize);
   numBlock = (second_half_N-1)/threadsPerBlock.x+1;

   cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY, first_half_bytes, nullptr, nullptr);
   cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY, first_half_bytes, nullptr, nullptr);
   cl_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, first_half_bytes, nullptr, nullptr);
   if (cl_a == nullptr || cl_b == nullptr || cl_c == nullptr) {
      std::cerr << "Create buffer failed" << std::endl;
      exit(-1);
   }

   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_a);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_b);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_c);
   err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &first_half_N);
   if (err != CL_SUCCESS) {
      std::cerr << "Set kernel arg failed: "+getErrorString(err) << std::endl;
      exit(-1);
   }

   CUDA_SAFE_CALL(cudaMalloc((void**) &cuda_a, second_half_bytes));
   CUDA_SAFE_CALL(cudaMalloc((void**) &cuda_b, second_half_bytes));
   CUDA_SAFE_CALL(cudaMalloc((void**) &cuda_c, second_half_bytes));
   timer_stop('u');
}


void vecAdd::heterogeneous_overlap_routine(){

   timer_start("heterogeneous_overlap_routine_memcpy_h2d_time",'u');
   // CUDA memcpy
   CUDA_SAFE_CALL(cudaMemcpyAsync(cuda_a, h_a+first_half_N, second_half_bytes, cudaMemcpyHostToDevice,cuda_stream));
   CUDA_SAFE_CALL(cudaMemcpyAsync(cuda_b, h_b+first_half_N, second_half_bytes, cudaMemcpyHostToDevice,cuda_stream));
   CUDA_SAFE_CALL(cudaMemset(cuda_c, 0, second_half_bytes));

   // OpenCL memcpy
   err = clEnqueueWriteBuffer(queue, cl_a, CL_FALSE, 0, first_half_bytes, h_a, 0, nullptr, nullptr);
   err |= clEnqueueWriteBuffer(queue, cl_b, CL_FALSE, 0, first_half_bytes, h_b, 0, nullptr, nullptr);
   if (err != CL_SUCCESS) {
      std::cerr << "Enqueue Write Buffer failed" << std::endl;
      exit(-1);
   }
   // Fence 1
   cudaStreamSynchronize(cuda_stream);
   clFinish(queue);
   timer_stop('u');

   timer_start("heterogeneous_overlap_routine_kernel_time",'u');
   // Launch OpenCL Kernel
   err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize,
                                0, nullptr, nullptr);
   // Launch CUDA Kernel
   vec_add_kernel<<<numBlock, threadsPerBlock, 0, cuda_stream>>>(cuda_a, cuda_b, cuda_c, second_half_N);


   if (err != CL_SUCCESS) {
      std::cerr << "Run kernel failed" << std::endl;
      exit(-1);
   }

   // Fence 2
   // clFinish(queue);
   CUDA_SAFE_CALL(cudaDeviceSynchronize());
   timer_stop('u');

   timer_start("heterogeneous_overlap_routine_memcpy_d2h_time",'u');
   // CUDA memcpy
   CUDA_SAFE_CALL(cudaMemcpyAsync(h_c+first_half_N, cuda_c, second_half_bytes, cudaMemcpyDeviceToHost, cuda_stream));

   // OpenCL memcpy
   err = clEnqueueReadBuffer(queue, cl_c, CL_FALSE, 0, first_half_bytes, h_c, 0, nullptr, nullptr);

   if (err != CL_SUCCESS) {
      std::cerr << "Read data failed" << std::endl;
      exit(-1);
   }

   // Fence 3
   clFinish(queue);
   cudaStreamSynchronize(cuda_stream);
   timer_stop('u');
}


void vecAdd::verify_result(const std::string& name) {
   float sum = 0;
   for (int i = 0; i < N; i++)
      sum += h_c[i];
   std::cout << "Result on " + name + " : " << sum << std::endl;
}


void vecAdd::output_reset() {
   memset(h_c, 0, bytes);
}