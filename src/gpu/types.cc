#include "types.h"

#ifndef ENABLE_GPU

void checkCudaErrors(int code) { assert(code == 0); }
int cudaSetDevice(int dev) { return 0; }
int cudaFree(void *ptr) {
  free(ptr);
  return 0;
}
int cudaMalloc(void **tmp, size_t num_bytes) {
  *tmp = malloc(num_bytes);
  return 0;
}
int cudaDeviceEnablePeerAccess(int dev1, int dev2) { return 0; }
int cudaStreamCreate(int *tmp) { return 0; }
int cublasCreate(int *tmp) { return 0; }
int cublasSetStream(int tmp, int tmp2) { return 0; }
int cudaStreamSynchronize(cudaStream_t) { return 0; }
int cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                        int srcDevice, size_t count, cudaStream_t stream) {
  memcpy(dst, src, count);
  return 0;
}
int cudaMemcpy(void *dst, const void *src, size_t count, int kind) {
  memcpy(dst, src, count);
  return 0;
}
int cudaFreeAsync(void *me, cudaStream_t) {
  free(me);
  return 0;
}

#endif