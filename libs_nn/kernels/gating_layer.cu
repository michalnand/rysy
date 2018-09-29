#include "gating_layer.cuh"


#define SIGMOID(x) (1.0/(1.0 + exp(-x)))
#define SIGMOID_DERIVATIVE(x) (SIGMOID(x)*(1.0 - SIGMOID(x)))



__host__
void cpu_gating_forward_kernel(float *output, float *input, unsigned int size)
{
  for (unsigned int idx = 0; idx < size; idx++)
  {
    unsigned int idx_input  = idx + 0*size;
    unsigned int idx_gate   = idx + 1*size;

    float g = SIGMOID(input[idx_gate]);

    float tmp = input[idx_input]*g;

    output[idx] = tmp;
  }
}

__global__
void cuda_gating_forward_kernel(float *output, float *input, unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    unsigned int idx_input  = idx + 0*size;
    unsigned int idx_gate   = idx + 1*size;

    float g = SIGMOID(input[idx_gate]);

    float tmp = input[idx_input]*g;

    output[idx] = tmp;
  }
}

void gating_layer_forward(  Tensor &output, Tensor &input)
{
  unsigned int size = output.size();

  #ifdef NETWORK_USE_CUDA

    unsigned int block_size = 16;
    if (size > 256)
      block_size = 256;

    dim3 block(block_size);
    dim3 grid((size + block.x - 1)/block.x);

    cuda_gating_forward_kernel<<<grid, block>>>(output.v, input.v, size);
    cudaDeviceSynchronize();

  #else

    cpu_gating_forward_kernel(output.v, input.v, size);

  #endif
}


__host__
void cpu_gating_backward_kernel(float *error_back, float *input, float *error, float *output, unsigned int size)
{
  for (unsigned int idx = 0; idx < size; idx++)
  {
      unsigned int idx_input   = idx;
      unsigned int idx_gate    = idx + size;

      float w0 = input[idx_input];
      float w1 = SIGMOID(input[idx_gate]);

      error_back[idx_input] = error[idx_input]*w1;
      error_back[idx_gate]  = SIGMOID_DERIVATIVE(output[idx])*error[idx_input]*w0;
  }
}

__global__
void cuda_gating_backward_kernel(float *error_back, float *input, float *error, float *output, unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    unsigned int idx_input   = idx;
    unsigned int idx_gate    = idx + size;

    float w0 = input[idx_input];
    float w1 = SIGMOID(input[idx_gate]);

    error_back[idx_input] = error[idx_input]*w1;
    error_back[idx_gate]  = SIGMOID_DERIVATIVE(output[idx])*error[idx_input]*w0;
  } 
}

void gating_layer_backward(Tensor &error_back, Tensor &input, Tensor &output, Tensor &error)
{
  unsigned int size = output.size();

  #ifdef NETWORK_USE_CUDA

      unsigned int block_size = 16;
      if (size >= 256)
        block_size = 256;

      dim3 block(block_size);
      dim3 grid((size + block.x - 1)/block.x);

      cuda_gating_backward_kernel<<<grid, block>>>(error_back.v, input.v, error.v, output.v, size);
      cudaDeviceSynchronize();
  #else

    cpu_gating_backward_kernel(error_back.v, input.v, error.v, output.v, size);

  #endif
}
