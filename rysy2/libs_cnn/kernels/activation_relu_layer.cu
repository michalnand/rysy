#include "activation_relu_layer.cuh"

__host__
void cpu_activation_relu_forward_kernel(float *output, float *input, unsigned int size)
{
    for (unsigned int idx = 0; idx < size; idx++)
    {
        float tmp = input[idx];

        if (tmp < 0.0)
            tmp = 0.0;

        output[idx] = tmp;
    }
}

__global__
void cuda_activation_relu_forward_kernel(float *output, float *input, unsigned int size)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < size)
    {
        float tmp = input[idx];

        if (tmp < 0.0)
            tmp = 0.0;

        output[idx] = tmp;
    }
}

void activation_relu_layer_forward(  Tensor &output, Tensor &input)
{
    unsigned int size = output.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(16);
        dim3 grid((size + block.x + 1)/block.x);

        cuda_activation_relu_forward_kernel<<<grid, block>>>(output.v, input.v, size);
        cudaDeviceSynchronize();

    #else

        cpu_activation_relu_forward_kernel(output.v, input.v, size);

    #endif
}


__host__
void cpu_activation_relu_backward_kernel(float *error_back, float *error, float *output, unsigned int size)
{
  for (unsigned int idx = 0; idx < size; idx++)
  {
    if (output[idx] > 0.0)
      error_back[idx] = error[idx];
    else
      error_back[idx] = 0.0;
  }
}

__global__
void cuda_activation_relu_backward_kernel(float *error_back, float *error, float *output, unsigned int size)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < size)
    {
        if (output[idx] > 0.0)
            error_back[idx] = error[idx];
        else
            error_back[idx] = 0.0;
    }
}

void activation_relu_layer_backward( Tensor &error_back, Tensor &output, Tensor &error)
{
    unsigned int size = output.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(16);
        dim3 grid((size + block.x + 1)/block.x);

        cuda_activation_relu_backward_kernel<<<grid, block>>>(error_back.v, error.v, output.v, size);
        cudaDeviceSynchronize();
    #else

        cpu_activation_relu_backward_kernel(error_back.v, error.v, output.v, size);

    #endif
}
