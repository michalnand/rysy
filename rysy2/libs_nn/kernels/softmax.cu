#include "softmax.cuh"


__host__
void cpu_softmax_forward_kernel(    float *output,
                                    float *input,
                                    unsigned int size)
{
    float max = input[0];

    for (unsigned int i = 0; i < size; i++)
    {
        if (input[i] > max)
            max = input[i];
    }

    float sum = 0.0;
    for (unsigned int i = 0; i < size; i++)
        sum+= exp(input[i] - max);

    for (unsigned int i = 0; i < size; i++)
        output[i] = exp(input[i] - max)/sum;
}


__global__
void cuda_softmax_forward_kernel(   float *output,
                                    float *input,
                                    unsigned int size)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < size)
    {
        float max = input[0];

        for (unsigned int i = 0; i < size; i++)
        {
            if (input[i] > max)
                max = input[i];
        }

        float sum = 0.0;
        for (unsigned int i = 0; i < size; i++)
            sum+= exp(input[i] - max);

        output[idx] = exp(input[idx] - max)/sum;
    }
}

void softmax_layer_forward( Tensor &output, Tensor &input)
{
    #ifdef NETWORK_USE_CUDA

        dim3 block(8);
        dim3 grid(input.size());

        cuda_softmax_forward_kernel<<<grid, block>>>(output.v, input.v, input.size());
        cudaDeviceSynchronize();

    #else

        cpu_softmax_forward_kernel(output.v, input.v, input.size());

    #endif
}












__host__
void cpu_softmax_backward_kernel(   float *error_back,
                                    float *output,
                                    float *error,
                                    unsigned int size)
{

    for (unsigned int i = 0; i < size; i++)
        error_back[i] = output[i]*(1.0 - output[i])*error[i];
}


__global__
void cuda_softmax_backward_kernel(  float *error_back,
                                    float *output,
                                    float *error,
                                    unsigned int size)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < size)
    {
        error_back[idx] = output[idx]*(1.0 - output[idx])*error[idx];
    }
}

void softmax_layer_backward(    Tensor &error_back,
                                Tensor &output,
                                Tensor &error)
{
    #ifdef NETWORK_USE_CUDA

        dim3 block(8);
        dim3 grid(error_back.size());

        cuda_softmax_backward_kernel<<<grid, block>>>(error_back.v, output.v, error.v, error_back.size());
        cudaDeviceSynchronize();

    #else

        cuda_softmax_backward_kernel(error_back.v, output.v, error.v, error_back.size());

    #endif
}
