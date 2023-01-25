
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <mkl.h>
#include <mkl_cblas.h>
#include <sys/mman.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>  
#include <cublas_v2.h>
#include "../third_party/cuda/gpu.h"

using namespace std::chrono;

int main() {

  size_t repeat = 200;
  std::vector<size_t> sizes = {10, 1000, 2000, 3000};

  for(auto num_bytes : sizes) {
    auto start = high_resolution_clock::now();
    for(auto idx = 0; idx < repeat; ++idx) {
        float *in1DataGPU;
        // checkCudaErrors(cudaMallocHost(&in1DataGPU, num_bytes * num_bytes * sizeof(float)));
        in1DataGPU = (float*)malloc(num_bytes * num_bytes * sizeof(float));
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);
    std::cout << "Alloc was" << 1e-9f * duration.count() << "s" << std::endl;
  }

}