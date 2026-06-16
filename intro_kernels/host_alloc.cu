#include "cuda.h"
#include <stdio.h>

float cuda_malloc_test(int size, bool up) {
  cudaEvent_t start, stop;
  int *a, *dev_a;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  a = (int *)malloc(size *sizeof(*a));
  cudaMalloc((void**)&dev_a, size * sizeof(*dev_a));

  cudaEventRecord(start, 0);
  for (int i = 0; i < 100; i++) {
    if (up) {
      cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice);
    }
    else {
      cudaMemcpy(a, dev_a, size * sizeof(*a), cudaMemcpyDeviceToHost);
    }
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  free(a);
  cudaFree(dev_a);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsedTime;
}