#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>

const int N = 4 * 1024 * 1024;
const int nStreams = 4;

__global__ void init(float *a, int offset, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[offset + i] = 1.0f;
  }
}

float timeSequential(float *dev_a, float *a) {
  cudaEvent_t start, stop;
  float elapsedTime;
  int chunk = N / nStreams;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  for (int i = 0; i < nStreams; i++) {
    int offset = i * chunk;
    cudaMemcpy(dev_a + offset, a + offset, chunk * sizeof(float), cudaMemcpyHostToDevice);
    init<<<(chunk + 255) / 256, 256>>>(dev_a, offset, chunk);
    cudaMemcpy(a + offset, dev_a + offset, chunk * sizeof(float), cudaMemcpyDeviceToHost);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsedTime;
}

float timeStreamed(float *dev_a, float *a, cudaStream_t *streams) {
  cudaEvent_t start, stop;
  float elapsedTime;
  int chunk = N / nStreams;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  for (int i = 0; i < nStreams; i++) {
    int offset = i * chunk;
    cudaMemcpyAsync(dev_a + offset, a + offset, chunk * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    init<<<(chunk + 255) / 256, 256, 0, streams[i]>>>(dev_a, offset, chunk);
    cudaMemcpyAsync(a + offset, dev_a + offset, chunk * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
  }
  for (int i = 0; i < nStreams; i++) {
    cudaStreamSynchronize(streams[i]);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsedTime;
}

int main(void) {
  float *a, *dev_a;
  cudaStream_t streams[nStreams];

  a = (float *)malloc(N * sizeof(float));
  cudaMalloc((void **)&dev_a, N * sizeof(float));

  for (int i = 0; i < nStreams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  float sequential = timeSequential(dev_a, a);
  float streamed = timeStreamed(dev_a, a, streams);

  printf("Sequential (default stream): %f ms\n", sequential);
  printf("Streamed (%d streams):       %f ms\n", nStreams, streamed);

  for (int i = 0; i < nStreams; i++) {
    cudaStreamDestroy(streams[i]);
  }
  free(a);
  cudaFree(dev_a);
  return 0;
}
