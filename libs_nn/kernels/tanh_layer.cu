#include "tanh_layer.cuh"

#define TANH(x)             (tanh(x))
#define TANH_DERIVATIVE(x)  (1.0 - x*x)

__host__
void cpu_tanh_forward_kernel(float *output, float *input, unsigned int size)
{
  for (unsigned int idx = 0; idx < size; idx++)
  {
    float tmp = input[idx];
    output[idx] = TANH(tmp);
  }
}

__global__
void cuda_tanh_forward_kernel(float *output, float *input, unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    float tmp = input[idx];
    output[idx] = TANH(tmp);
  }
}

void tanh_layer_forward(  Tensor &output, Tensor &input)
{
  unsigned int size = output.size();

  #ifdef NETWORK_USE_CUDA

    unsigned int block_size = 16;
    if (size > 256)
      block_size = 256;

    dim3 block(block_size);
    dim3 grid((size + block.x - 1)/block.x);

    cuda_tanh_forward_kernel<<<grid, block>>>(output.v, input.v, size);
    cudaDeviceSynchronize();

  #else

    cpu_tanh_forward_kernel(output.v, input.v, size);

  #endif
}


__host__
void cpu_tanh_backward_kernel(float *error_back, float *error, float *output, unsigned int size)
{
  for (unsigned int idx = 0; idx < size; idx++)
  {
    error_back[idx] = error[idx]*TANH_DERIVATIVE(output[idx]);
  }
}

__global__
void cuda_tanh_backward_kernel(float *error_back, float *error, float *output, unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    error_back[idx] = error[idx]*TANH_DERIVATIVE(output[idx]);
  }
}

void tanh_layer_backward( Tensor &error_back, Tensor &output, Tensor &error)
{
  unsigned int size = output.size();

  #ifdef NETWORK_USE_CUDA

      unsigned int block_size = 16;
      if (size >= 256)
        block_size = 256;

      dim3 block(block_size);
      dim3 grid((size + block.x - 1)/block.x);

      cuda_tanh_backward_kernel<<<grid, block>>>(error_back.v, error.v, output.v, size);
      cudaDeviceSynchronize();
  #else

    cpu_tanh_backward_kernel(error_back.v, error.v, output.v, size);

  #endif
}
