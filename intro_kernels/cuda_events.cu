#include "cuda.h"
#include <stdio.h>

__global__ void heavyKernel(void) {
  volatile int x = 0;
  for (int i = 0; i < 100000000; i++) {
    x++;
  }
}

int main(void) {
  cudaEvent_t start;
  cudaEvent_t stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  heavyKernel<<<1, 1>>>();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); // to remain reliable despite async cpu execution
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("GPU elapsed time: %f ms\n", elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
