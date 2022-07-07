
#include <chrono>
#include <cstddef>
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
  std::vector<size_t> sizes = {10, 1000};
  std::vector<std::vector<float*>> allocs;

  float *tmp;
  checkCudaErrors(cudaMallocHost(&tmp, sizes.back() * sizes.back() * sizeof(float)));
  for(auto idx = 0; idx < sizes.back() * sizes.back(); ++idx) {
    tmp[idx] = 1.0f;
  }

  cudaStream_t cpy_stream;
  checkCudaErrors(cudaStreamCreate(&cpy_stream));

  for(auto size : sizes) {
    for(auto idx = 0; idx < repeat; ++idx) {
        float *in1DataGPU;
        checkCudaErrors(cudaMalloc(&in1DataGPU, size * size * sizeof(float)));

        float *in2DataGPU;
        checkCudaErrors(cudaMalloc(&in2DataGPU, size * size * sizeof(float)));

        float *outGPU;
        checkCudaErrors(cudaMalloc(&outGPU, size * size * sizeof(float)));

        allocs.push_back({in1DataGPU, in2DataGPU, outGPU});

        checkCudaErrors(cudaMemcpyAsync(in1DataGPU, tmp, 
                        size * size * sizeof(float), 
                        cudaMemcpyHostToDevice, cpy_stream));
        checkCudaErrors(cudaMemcpyAsync(in2DataGPU, tmp, 
                        size * size * sizeof(float), 
                        cudaMemcpyHostToDevice, cpy_stream));
        checkCudaErrors(cudaStreamSynchronize(cpy_stream));
    }
  }


  // init the stream
  cudaStream_t run_stream;
  checkCudaErrors(cudaStreamCreate(&run_stream));

  // create the cublas handle
  cublasHandle_t cublas_handle;
  checkCudaErrors(cublasCreate(&cublas_handle));
  checkCudaErrors(cublasSetStream(cublas_handle, run_stream));

  size_t jdx = 0;
  for(auto size : sizes) {

    auto start = high_resolution_clock::now();
    for(auto idx = 0; idx < repeat; ++idx) {

        float* in1DataGPU = allocs[jdx][0];
        float* in2DataGPU = allocs[jdx][1];
        float* outDataGPU = allocs[jdx][2];

        float alpha=1.0f;                                             
        float beta=0.0f;                                              
        checkCudaErrors(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha, in1DataGPU, size, in2DataGPU, size, &beta, outDataGPU, size));
        

        jdx++;
    }
    checkCudaErrors(cudaStreamSynchronize(run_stream));

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Kernel was" << 1e-6f * duration.count() << "s" << std::endl;
  }

}