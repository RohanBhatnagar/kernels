#include <stdio.h>
#define imin(a,b) (a < b ? a : b)
const int N = 33 * 1024;
const int threadsPerBlock = 256;

__global__ void dot(float *a, float *b, float *c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIdx = threadIdx.x;

  __shared__ float cache[threadsPerBlock];

  float temp = 0;
  while (tid < N) {
    temp += a[tid] * b[tid];
    tid += blockDim.x + gridDim.x;
  }
  cache[cacheIdx] = temp;
  __syncthreads(); // All threads will reach this point before next instruction.

  // Reduction
  int i = blockDim.x / 2;
  while (i != 0) {
    if (cacheIdx < i) {
      cache[cacheIdx] += cache[cacheIdx + i];
    }
    __syncthreads();
    i /= 2;
  }

  if (cacheIdx == 0) {
    c[blockIdx.x] = cache[0];
  }
}

const int blocksPerGrid = imin(32, N + threadsPerBlock - 1 / threadsPerBlock);

int main(void) {
  float *a, *b, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;

  partial_c = new float[blocksPerGrid];

  cudaMalloc((void **)&dev_a, sizeof(float) * N);
  cudaMalloc((void **)&dev_b, sizeof(float) * N);
  cudaMalloc((void **)&dev_partial_c, sizeof(float) * blocksPerGrid);

  a = new float[N];
  b = new float[N];

  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = 2*i;
  }

  cudaMemcpy(dev_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

  dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

  cudaMemcpy(partial_c, dev_partial_c, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost);

  for (int i = 0; i < blocksPerGrid; i++) {
    printf("partial_c[%d] = %f\n", i, partial_c[i]);
  }

  delete[] partial_c;

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_partial_c);

  return 0;

}