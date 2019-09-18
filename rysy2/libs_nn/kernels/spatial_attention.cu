#include "spatial_attention.cuh"


#define SIGMOID(x)                  (1.0/(1.0 + exp(-x)))
#define SIGMOID_DERIVATIVE(y)       (y*(1.0 - y))



__host__
void cpu_spatial_attention_forward_kernel(  float *output,
                                            float *input,
                                            float *input_attention,
                                            unsigned int width,
                                            unsigned int height,
                                            unsigned int channels)
{
    for (unsigned int y = 0; y < height; y++)
    for (unsigned int x = 0; x < width; x++)
    {
        unsigned int input_idx = y*width + x;

        float attention = SIGMOID(input_attention[input_idx]);

        for (unsigned int ch = 0; ch < channels; ch++)
        {
            output[input_idx] = input[input_idx]*(attention + 1.0);
            input_idx+= width*height;
        }
    }
}

__global__
void cuda_spatial_attention_forward_kernel( float *output,
                                            float *input,
                                            float *input_attention,
                                            unsigned int width,
                                            unsigned int height,
                                            unsigned int channels)
{
    unsigned int x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (y < height)
    if (x < width)
    {
        unsigned int input_idx = y*width + x;

        float attention = SIGMOID(input_attention[input_idx]);

        for (unsigned int ch = 0; ch < channels; ch++)
        {
            output[input_idx] = input[input_idx]*(attention + 1.0);
            input_idx+= width*height;
        }
    }
}

void spatial_attention_forward(  Tensor &output,
                                 Tensor &input,
                                 Tensor &input_attention)
{
    #ifdef NETWORK_USE_CUDA

        dim3 block(8, 8);
        dim3 grid((input.w() + block.x + 1)/block.x, (input.h() + block.y + 1)/block.y);

        cuda_spatial_attention_forward_kernel<<<grid, block>>>(output.v, input.v, input_attention.v, input.w(), input.h(), input.d());
        cudaDeviceSynchronize();

    #else

        cpu_spatial_attention_forward_kernel(output.v, input.v, input_attention.v, input.w(), input.h(), input.d());

    #endif
}


__host__
void cpu_spatial_attention_backward_kernel( float *error_back,
                                            float *error_back_attention,
                                            float *input,
                                            float *input_attention,
                                            float *error,

                                            unsigned int width,
                                            unsigned int height,
                                            unsigned int channels)
{
    for (unsigned int y = 0; y < height; y++)
    for (unsigned int x = 0; x < width; x++)
    {
        unsigned int input_idx = y*width + x;

        float attention = SIGMOID(input_attention[input_idx]);

        float attention_error_back_sum = 0.0;

        for (unsigned int ch = 0; ch < channels; ch++)
        {
            error_back[input_idx]    =  error[input_idx]*(attention + 1.0);
            attention_error_back_sum+=  error[input_idx]*input[input_idx]*SIGMOID_DERIVATIVE(attention);
            input_idx+= width*height;
        }

        error_back_attention[y*width + x] = attention_error_back_sum;
    }
}

__global__
void cuda_spatial_attention_backward_kernel(    float *error_back,
                                                float *error_back_attention,
                                                float *input,
                                                float *input_attention,
                                                float *error,

                                                unsigned int width,
                                                unsigned int height,
                                                unsigned int channels)
{
    unsigned int x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (y < height)
    if (x < width)
    {
        unsigned int input_idx = y*width + x;

        float attention = SIGMOID(input_attention[input_idx]);

        float attention_error_back_sum = 0.0;

        for (unsigned int ch = 0; ch < channels; ch++)
        {
            error_back[input_idx]    =  error[input_idx]*(attention + 1.0);
            attention_error_back_sum+=  error[input_idx]*input[input_idx]*SIGMOID_DERIVATIVE(attention);
            input_idx+= width*height;
        }

        error_back_attention[y*width + x] = attention_error_back_sum;
    }
}

void spatial_attention_backward(  Tensor &error_back,
                                  Tensor &error_back_attention,
                                  Tensor &input,
                                  Tensor &input_attention,
                                  Tensor &error)
{
    #ifdef NETWORK_USE_CUDA

        dim3 block(8, 8);
        dim3 grid((input.w() + block.x + 1)/block.x, (input.h() + block.y + 1)/block.y);


        cuda_spatial_attention_backward_kernel<<<grid, block>>>(error_back.v, error_back_attention.v, input.v, input_attention.v, error.v, input.w(), input.h(), input.d());
        cudaDeviceSynchronize();

    #else

        cpu_spatial_attention_backward_kernel(error_back.v, error_back_attention.v, input.v, input_attention.v, error.v, input.w(), input.h(), input.d());

    #endif
}
