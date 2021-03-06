#include "leaky_relu_layer.cuh"

#define LEAKY_RELU_CONST ((float)1.0/100.0)

__host__
void cpu_leaky_relu_forward_kernel(float *output, float *input, unsigned int size)
{
  for (unsigned int idx = 0; idx < size; idx++)
  {
    float tmp = input[idx];

    if (tmp < 0.0)
        tmp = LEAKY_RELU_CONST*tmp;

    output[idx] = tmp;
  }
}

__global__
void cuda_leaky_relu_forward_kernel(float *output, float *input, unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    float tmp = input[idx];

    if (tmp < 0.0)
        tmp = LEAKY_RELU_CONST*tmp;

    output[idx] = tmp;
  }
}

void leaky_relu_layer_forward(  Tensor &output, Tensor &input)
{
  unsigned int size = output.size();

  #ifdef NETWORK_USE_CUDA
    dim3 block(32);
    dim3 grid((size + block.x + 1)/block.x);

    cuda_leaky_relu_forward_kernel<<<grid, block>>>(output.v, input.v, size);
    cudaDeviceSynchronize();

  #else

    cpu_leaky_relu_forward_kernel(output.v, input.v, size);

  #endif
}


__host__
void cpu_leaky_relu_backward_kernel(float *error_back, float *error, float *output, unsigned int size)
{
  for (unsigned int idx = 0; idx < size; idx++)
  {
    if (output[idx] > 0.0)
      error_back[idx] = error[idx];
    else
      error_back[idx] = LEAKY_RELU_CONST*error_back[idx];
  }
}

__global__
void cuda_leaky_relu_backward_kernel(float *error_back, float *error, float *output, unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    if (output[idx] > 0.0)
      error_back[idx] = error[idx];
    else
      error_back[idx] = LEAKY_RELU_CONST*error_back[idx];
  }
}

void leaky_relu_layer_backward( Tensor &error_back, Tensor &output, Tensor &error)
{
  unsigned int size = output.size();

  #ifdef NETWORK_USE_CUDA
      dim3 block(32);
      dim3 grid((size + block.x + 1)/block.x);

      cuda_leaky_relu_backward_kernel<<<grid, block>>>(error_back.v, error.v, output.v, size);
      cudaDeviceSynchronize();
  #else

    cpu_leaky_relu_backward_kernel(error_back.v, error.v, output.v, size);

  #endif
}
