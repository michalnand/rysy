#include "spatial_attention.cuh"


#define SIGMOID(x)                  (1.0/(1.0 + exp(-x)))
#define SIGMOID_DERIVATIVE(x)       (SIGMOID(x)*(1.0 - SIGMOID(x)))



__host__
void cpu_spatial_attention_forward_kernel(  float *output,
                                            float *input,
                                            unsigned int size)
{
    for (unsigned int idx = 0; idx < size; idx++)
    {
        float activation = SIGMOID(input[idx]);

        float result = input[idx + size]*(activation + 1.0);

        output[idx] = result;
    }
}

__global__
void cuda_spatial_attention_forward_kernel( float *output,
                                            float *input,
                                            unsigned int size)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < size)
    {
        float activation = SIGMOID(input[idx]);

        float result = input[idx + size]*(activation + 1.0);

        output[idx] = result;
    }
}

void spatial_attention_forward(  Tensor &output,
                                 Tensor &input)
{
    unsigned int size = output.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(16);
        dim3 grid((size + block.x + 1)/block.x);

        cuda_spatial_attention_forward_kernel<<<grid, block>>>(output.v, input.v, size);
        cudaDeviceSynchronize();

    #else

        cpu_spatial_attention_forward_kernel(output.v, input.v, size);

    #endif
}


__host__
void cpu_spatial_attention_backward_kernel( float *error_back,
                                            float *input,
                                            float *error,

                                            unsigned int size)
{
    for (unsigned int idx = 0; idx < size; idx++)
    {
        error_back[idx]        = error[idx]*SIGMOID_DERIVATIVE(input[idx])*input[idx + size];
        error_back[idx+size]   = error[idx]*(SIGMOID(input[idx]) + 1.0);
    }
}

__global__
void cuda_spatial_attention_backward_kernel(    float *error_back,
                                                float *input,
                                                float *error,

                                                unsigned int size)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < size)
    {
        error_back[idx]        = error[idx]*SIGMOID_DERIVATIVE(input[idx])*input[idx + size];
        error_back[idx+size]   = error[idx]*(SIGMOID(input[idx]) + 1.0);
    }
}

void spatial_attention_backward( Tensor &error_back,
                                 Tensor &input,
                                 Tensor &error)
{
    unsigned int size = error.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(16);
        dim3 grid((size + block.x + 1)/block.x);

        cuda_spatial_attention_backward_kernel<<<grid, block>>>(error_back.v, input.v, error.v, size);
        cudaDeviceSynchronize();

    #else

        cpu_spatial_attention_backward_kernel(error_back.v, input.v, error.v, size);

    #endif
}
