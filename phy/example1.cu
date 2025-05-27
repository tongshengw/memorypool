// from: https://github.com/JuliaGPU/CUDA.jl/issues/2450

#include <stdio.h>
typedef double (*func)(double x);

__device__ double func1(double x)
{
return x+1.0f;
}

__device__ double func2(double x)
{
return x*2.0f;
}

__device__ func pfunc1 = func1;
__device__ func pfunc2 = func2;

__global__ void test_kernel(func* f, int n)
{
  double x = 1.0;

  for(int i=0;i<n;++i){
   x=f[i](x);
   printf("%g\n",x);
  }
}

int main(void)
{
  int N = 2;

  func* h_f;
  func* d_f;

  h_f = (func*)malloc(N*sizeof(func));

  cudaMalloc((void**)&d_f,N*sizeof(func));

  cudaMemcpyFromSymbol( &h_f[0], pfunc1, sizeof(func));
  cudaMemcpyFromSymbol( &h_f[1], pfunc2, sizeof(func));

  cudaMemcpy(d_f,h_f,N*sizeof(func),cudaMemcpyHostToDevice);

  test_kernel<<<1,1>>>(d_f,N);

  cudaFree(d_f);
  free(h_f);

  return 0;
}

