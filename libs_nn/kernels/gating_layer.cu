#include "gating_layer.cuh"


#define SIGMOID(x) (1.0/(1.0 + exp(-x)))

#define SIGMOID_DERIVATIVE(x) (SIGMOID(x)*(1.0 - SIGMOID(x)))



__host__
void cpu_gating_forward_kernel(float *output, float *input, unsigned int size)
{
  for (unsigned int idx = 0; idx < size; idx++)
  {
    unsigned int idx_input  = idx;
    unsigned int idx_gate   = idx + size;

    output[idx] = input[idx_gate]*input[idx_input];

  //  output[idx] = SIGMOID(input[idx_gate])*input[idx_input];
  }
}

__global__
void cuda_gating_forward_kernel(float *output, float *input, unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    unsigned int idx_input  = idx;
    unsigned int idx_gate   = idx + size;

    output[idx] = input[idx_gate]*input[idx_input];

    // output[idx] = SIGMOID(input[idx_gate])*input[idx_input];
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

      float err = error[idx];

      /*
      error_back[idx_input] = err*SIGMOID(input[idx_gate]);
      error_back[idx_gate]  = err*SIGMOID_DERIVATIVE(input[idx_gate])*input[idx_input];
      */

      error_back[idx_input] = err*input[idx_gate];
      error_back[idx_gate]  = err*input[idx_input]; 
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

    float err = error[idx];

    /*
    error_back[idx_input] = err*SIGMOID(input[idx_gate]);
    error_back[idx_gate]  = err*SIGMOID_DERIVATIVE(input[idx_gate])*input[idx_input];
    */

    error_back[idx_input] = err*input[idx_gate];
    error_back[idx_gate]  = err*input[idx_input];
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
