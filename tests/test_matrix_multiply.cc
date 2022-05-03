
#include <chrono>
#include <iostream>
#include <mkl.h>
#include <mkl_cblas.h>
#include <sys/mman.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>  
#include <cublas_v2.h>

using namespace std::chrono;

int main() {

  int N = 24000;

  float *in1Data;
  cudaMallocHost(&in1Data, N * N * sizeof(float));
  float *in2Data;
  cudaMallocHost(&in2Data, N * N * sizeof(float));
  float *outData;
  cudaMallocHost(&outData, N * N * sizeof(float));

  float *in1DataGPU;
  cudaMalloc(&in1DataGPU, N * N * sizeof(float));
  float *in2DataGPU;
  cudaMalloc(&in2DataGPU, N * N * sizeof(float));
  float *outDataGPU;
  cudaMalloc(&outDataGPU, N * N * sizeof(float));

  // make the random stream
  VSLStreamStatePtr stream;
  vslNewStream(&stream, VSL_BRNG_MCG31, 123);

  // the left and right boundary
  auto left = 0.0f;
  auto right = 1.0f;

  // create a bunch of random numbers
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, (int32_t)(N * N), in1Data,
               left, right);
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, (int32_t)(N * N), in2Data,
               left, right);

  cudaStream_t cpy_stream;
  cudaStreamCreate(&cpy_stream);

  // delete the stream
  vslDeleteStream(&stream);

  auto start_copy = high_resolution_clock::now();
  cudaMemcpyAsync(in1DataGPU, in1Data, 
                  N * N * sizeof(float), 
                  cudaMemcpyHostToDevice, cpy_stream);
  cudaMemcpyAsync(in2DataGPU, in2Data, 
                  N * N * sizeof(float), 
                  cudaMemcpyHostToDevice, cpy_stream);
  cudaStreamSynchronize(cpy_stream);
  auto stop_copy = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop_copy - start_copy);
  std::cout << "Copy was" << 1e-6f * duration.count() << "s" << std::endl;

  // init the stream
  cudaStream_t run_stream;
  cudaStreamCreate(&run_stream);

  // create the cublas handle
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cublasSetStream(cublas_handle, run_stream);

  auto start = high_resolution_clock::now();
  float alpha=1.0f;                                             
  float beta=0.0f;                                              
  cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, in1DataGPU, N, in2DataGPU, N, &beta, outDataGPU, N);
  cudaStreamSynchronize(run_stream);
  auto stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  std::cout << "Kernel was" << 1e-6f * duration.count() << "s" << std::endl;
}