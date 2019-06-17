#include "dropout_layer.cuh"

__host__
void cpu_dropout_forward_kernel(float *output, float *input, float *noise, float dropout, unsigned int size)
{
    for (unsigned int idx = 0; idx < size; idx++)
    {
        float tmp = (noise[idx] + 1.0)/2.0;

        if (tmp < dropout)
            output[idx] = 0.0;
        else
            output[idx] = input[idx];
    }
}

__global__
void cuda_dropout_forward_kernel(float *output, float *input, float *noise, float dropout, unsigned int size)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < size)
    {
        float tmp = (noise[idx] + 1.0)/2.0;

        if (tmp < dropout)
            output[idx] = 0.0;
        else
            output[idx] = input[idx];
    }
}

void dropout_layer_forward(  Tensor &output, Tensor &input, Tensor &noise, float dropout)
{
    unsigned int size = output.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(16);
        dim3 grid((size + block.x + 1)/block.x);

        cuda_dropout_forward_kernel<<<grid, block>>>(output.v, input.v, noise.v, dropout, size);
        cudaDeviceSynchronize();

    #else

        cpu_dropout_forward_kernel(output.v, input.v, noise.v, dropout, size);

    #endif
}


__host__
void cpu_dropout_backward_kernel(float *error_back, float *error, float *output, unsigned int size)
{
    for (unsigned int idx = 0; idx < size; idx++)
    {
        float tmp = output[idx];
        if (tmp < 0.0)
            tmp = -tmp;

        if (tmp > 0.000000001)
            error_back[idx] = error[idx];
        else
            error_back[idx] = 0.0;
    }
}

__global__
void cuda_dropout_backward_kernel(float *error_back, float *error, float *output, unsigned int size)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < size)
    {
        float tmp = output[idx];
        if (tmp < 0.0)
            tmp = -tmp;

        if (tmp > 0.000000001)
            error_back[idx] = error[idx];
        else
            error_back[idx] = 0.0;
    }
}

void dropout_layer_backward( Tensor &error_back, Tensor &output, Tensor &error)
{
    unsigned int size = output.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(16);
        dim3 grid((size + block.x + 1)/block.x);

        cuda_dropout_backward_kernel<<<grid, block>>>(error_back.v, error.v, output.v, size);
        cudaDeviceSynchronize();
    #else

        cpu_dropout_backward_kernel(error_back.v, error.v, output.v, size);

    #endif
}
