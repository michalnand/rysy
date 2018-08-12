#include "cuda_tensor.cuh"
#include <stdint.h>

__device__ __managed__ float g_cuda_result;



void cuda_tensor_clear(float *v, unsigned int size)
{
  cudaMemset(v, 0, size*sizeof(float));
}

__global__
void cuda_tensor_set_const_kernel(float *v, unsigned int size, float value)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
    v[idx] = value;
}

void cuda_tensor_set_const(float *v, unsigned int size, float value)
{
  dim3 block(16);
  dim3 grid((size + block.x - 1)/block.x);

  cuda_tensor_set_const_kernel<<<grid, block>>>(v, size, value);
  cudaDeviceSynchronize();
}

__device__ unsigned int cuda_g_rnd_a, cuda_g_rnd_b;

__global__
void cuda_tensor_random_kernel(float *v, unsigned int size, float range, unsigned int seed)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    unsigned int rnd_a = (seed + idx + cuda_g_rnd_a)*(uint32_t)1103515245 + (uint32_t)12345;
    atomicAdd(&cuda_g_rnd_a, rnd_a);

    unsigned int rnd_b = (seed + idx + cuda_g_rnd_b)*(uint32_t)48271;
    atomicAdd(&cuda_g_rnd_b, rnd_b);

    float rndf = (2.0*((rnd_a^rnd_b)%1000000)/1000000.0) - 1.0;
    v[idx] = range*rndf;
  }
}

void cuda_tensor_random(float *v, unsigned int size, float range)
{
  unsigned int seed = rand();

  dim3 block(16);
  dim3 grid((size + block.x - 1)/block.x);

  cuda_tensor_random_kernel<<<grid, block>>>(v, size, range, seed);
  cudaDeviceSynchronize();
}


__global__
void cuda_tensor_add_kernel(float *v, float *rhs, unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
    v[idx]+= rhs[idx];
}

void cuda_tensor_add(float *result, float *rhs, unsigned int size)
{
  dim3 block(16);
  dim3 grid((size + block.x - 1)/block.x);

  cuda_tensor_add_kernel<<<grid, block>>>(result, rhs, size);
  cudaDeviceSynchronize();
}


__global__
void cuda_tensor_sub_kernel(float *v, float *rhs, unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
    v[idx]-= rhs[idx];
}

void cuda_tensor_sub(float *result, float *rhs, unsigned int size)
{
  dim3 block(16);
  dim3 grid((size + block.x - 1)/block.x);

  cuda_tensor_sub_kernel<<<grid, block>>>(result, rhs, size);
  cudaDeviceSynchronize();
}


__global__
void cuda_tensor_mul_kernel(float *v, float value, unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
    v[idx]*= value;
}

void cuda_tensor_mul(float *result, float value, unsigned int size)
{
  dim3 block(16);
  dim3 grid((size + block.x - 1)/block.x);

  cuda_tensor_mul_kernel<<<grid, block>>>(result, value, size);
  cudaDeviceSynchronize();
}


void cuda_tensor_set_element(float *result, float value, unsigned int idx)
{
  cudaMemcpy(&result[idx], &value, sizeof(float), cudaMemcpyHostToDevice);
}

float cuda_tensor_get_element(float *src_ptr, unsigned int idx)
{
  float result = 0.0;

  cudaMemcpy(&result, &src_ptr[idx], sizeof(float), cudaMemcpyDeviceToHost);

  return result;
}

__global__
void cuda_tensor_regularization_l1_kernel(float *v, float lambda, unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    if (v[idx] > 0.0)
      v[idx]-= lambda;
    else
      v[idx]+= lambda;
  }
}

void cuda_tensor_regularization_l1(float *result, float lambda, unsigned int size)
{
  unsigned int block_size;
  
  if (size >= 256)
    block_size = 256;
  else
    block_size = 16;

  dim3 block(block_size);
  dim3 grid((block_size + block.x - 1)/block.x);

  cuda_tensor_regularization_l1_kernel<<<grid, block>>>(result, lambda, size);
  cudaDeviceSynchronize();
}



__global__
void cuda_rms_kernel(float *va, float *vb, unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < size)
  {
    float tmp = va[idx] - vb[idx];
    tmp = tmp*tmp;
    atomicAdd(&g_cuda_result, tmp);
  }
}


void cuda_rms(float *result, float *va, float *vb, unsigned int size)
{
  dim3 block(16);
  dim3 grid((size + block.x - 1)/block.x);

  g_cuda_result = 0.0;

  cuda_rms_kernel<<<grid, block>>>(va, vb, size);
  cudaDeviceSynchronize();

  *result = g_cuda_result;
}
