
#include "vecAdd.cuh"


int main(int argc, char* argv[]) {

   vecAdd vec_add(120000000,3,16,512);

   vec_add.init_host_data();

   timer_start("cpu_routine",'m');
   vec_add.cpu_routine(); vec_add.verify_result(CPU); vec_add.output_reset();
   timer_stop('m'); std::cout << std::endl;

   timer_start("opencl_routine",'m');
   vec_add.opencl_routine(); vec_add.verify_result(INTEL_GPU); vec_add.output_reset();
   timer_stop('m'); std::cout << std::endl;

   timer_start("cuda_routine",'m');
   vec_add.cuda_routine(); vec_add.verify_result(NVIDIA_GPU); vec_add.output_reset();
   timer_stop('m'); std::cout << std::endl;

   vec_add.heterogeneous_overlap_preprocess();
   timer_start("heterogeneous_overlap_routine",'m');
   vec_add.heterogeneous_overlap_routine(); vec_add.verify_result(HYBRID_GPU); vec_add.output_reset();
   timer_stop('m'); std::cout << std::endl;

   return 0;
}
