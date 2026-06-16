#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

__global__ void add(int a, int b, int *c) {
  *c = a + b;
}

int main() {
  int c;
  int *dev_c;

  cudaMalloc((void **)&dev_c, sizeof(int));
  add<<<1,1>>>(2, 7, dev_c);
  cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
  printf("c = %d\n", c);
  cudaFree(dev_c);
  return 0;
}