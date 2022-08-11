#include <thread>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>  
#include <cublas_v2.h>
#include <cublasLt.h>
#include "../third_party/cuda/gpu.h"
using namespace std::chrono;
const int N = 20000;

void set_to_one(float *blk) {
  for (size_t idx = 0; idx < N * N; ++idx) {
    blk[idx] = 1.0f;
  }
}

int main() {

  float *a_blk = (float *)malloc(sizeof(float) * N * N);
  set_to_one(a_blk);
  float *b_blk = (float *)malloc(sizeof(float) * N * N);
  set_to_one(b_blk);
  float *c_blk = (float *)malloc(sizeof(float) * N * N);

  auto start = high_resolution_clock::now();
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  
  float *a_gpu_blk, *b_gpu_blk, *c_gpu_blk;
  checkCudaErrors(cudaMalloc(&a_gpu_blk, N * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&b_gpu_blk, N * N * sizeof(float)));
  checkCudaErrors(cudaMalloc(&c_gpu_blk, N * N * sizeof(float)));
  cudaMemcpy(a_gpu_blk, a_blk, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaMemcpy(b_gpu_blk, b_blk, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  std::cout << "Time it took : " << (float)duration.count() * 1e-6f << std::endl;

  start = high_resolution_clock::now();
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, a_gpu_blk, N, b_gpu_blk, N, &beta, c_gpu_blk, N);
  cudaDeviceSynchronize();
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  std::cout << "Time it took for the kernel: " << (float)duration.count() * 1e-6f << std::endl;

  return 0;
}