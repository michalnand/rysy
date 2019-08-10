#include "activation_tanh_layer.cuh"

#define TANH(x)     (tanh(x))
#define TANH_D(y)   (1.0 - y*y)

__host__
void cpu_activation_tanh_forward_kernel(float *output, float *input, unsigned int size)
{
    for (unsigned int idx = 0; idx < size; idx++)
    {
        output[idx] = TANH(input[idx]);
    }
}

__global__
void cuda_activation_tanh_forward_kernel(float *output, float *input, unsigned int size)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < size)
    {
        output[idx] = TANH(input[idx]);
    }
}

void activation_tanh_layer_forward(  Tensor &output, Tensor &input)
{
    unsigned int size = output.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(16);
        dim3 grid((size + block.x + 1)/block.x);

        cuda_activation_tanh_forward_kernel<<<grid, block>>>(output.v, input.v, size);
        cudaDeviceSynchronize();

    #else

        cpu_activation_tanh_forward_kernel(output.v, input.v, size);

    #endif
}


__host__
void cpu_activation_tanh_backward_kernel(float *error_back, float *error, float *output, unsigned int size)
{
    for (unsigned int idx = 0; idx < size; idx++)
    {
        error_back[idx] = error[idx]*TANH_D(output[idx]);
    }
}

__global__
void cuda_activation_tanh_backward_kernel(float *error_back, float *error, float *output, unsigned int size)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < size)
    {
        error_back[idx] = error[idx]*TANH_D(output[idx]);
    }
}

void activation_tanh_layer_backward( Tensor &error_back, Tensor &output, Tensor &error)
{
    unsigned int size = output.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(16);
        dim3 grid((size + block.x + 1)/block.x);

        cuda_activation_tanh_backward_kernel<<<grid, block>>>(error_back.v, error.v, output.v, size);
        cudaDeviceSynchronize();

    #else

        cpu_activation_tanh_backward_kernel(error_back.v, error.v, output.v, size);

    #endif
}
