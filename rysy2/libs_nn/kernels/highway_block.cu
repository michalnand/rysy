#include "highway_block.cuh"

#define SIGMOID(x)                  (1.0/(1.0 + exp(-x)))
#define SIGMOID_DERIVATIVE(x)       (SIGMOID(x)*(1.0 - SIGMOID(x)))

#define ELU_ALPHA                   ((float)0.1)
#define ELU(x)                      ((x <= 0) ? (ELU_ALPHA*(exp(x) - 1.0)) : x)
#define ELU_DERIVATIVE(x)           ((x <= 0) ? (ELU_ALPHA*exp(x)) : 1.0)



__host__
void cpu_highway_forward_kernel(    float *output,
                                    float *input_t,
                                    float *input_h,
                                    float *input_c,

                                    unsigned int output_size)
{
    for (unsigned int idx = 0; idx < output_size; idx++)
    {
        float at = input_t[idx];
        float ah = ELU(input_h[idx]);
        float c  = SIGMOID(input_c[idx]);

        output[idx] = c*ah + (1.0 - c)*at;
    }
}


__global__
void cuda_highway_forward_kernel(   float *output,
                                    float *input_t,
                                    float *input_h,
                                    float *input_c,

                                    unsigned int output_size)
{
    unsigned int idx  = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < output_size)
    {
        float at = input_t[idx];
        float ah = ELU(input_h[idx]);
        float c  = SIGMOID(input_c[idx]);

        output[idx] = c*ah + (1.0 - c)*at;
    }
}



__host__
void cpu_highway_backward_kernel(   float *error_back_t,
                                    float *error_back_h,
                                    float *error_back_c,

                                    float *input_t,
                                    float *input_h,
                                    float *input_c,

                                    float *error,

                                    unsigned int output_size)
{
    for (unsigned int idx = 0; idx < output_size; idx++)
    {
        float at = input_t[idx];
        float ah = ELU(input_h[idx]);
        float c  = SIGMOID(input_c[idx]);

        error_back_t[idx] = error[idx]*(1.0 - c);
        error_back_h[idx] = error[idx]*ELU_DERIVATIVE(input_h[idx])*c;
        error_back_c[idx] = error[idx]*SIGMOID_DERIVATIVE(input_c[idx])*(c*ah + (1.0 - c)*at);
    }
}


__global__
void cuda_highway_backward_kernel(  float *error_back_t,
                                    float *error_back_h,
                                    float *error_back_c,

                                    float *input_t,
                                    float *input_h,
                                    float *input_c,

                                    float *error,

                                    unsigned int output_size)
{
    unsigned int idx  = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < output_size)
    {
        float at = input_t[idx];
        float ah = ELU(input_h[idx]);
        float c  = SIGMOID(input_c[idx]);

        error_back_t[idx] = error[idx]*(1.0 - c);
        error_back_h[idx] = error[idx]*ELU_DERIVATIVE(input_h[idx])*c;
        error_back_c[idx] = error[idx]*SIGMOID_DERIVATIVE(input_c[idx])*(c*ah + (1.0 - c)*at);
    }
}

void highway_layer_forward(Tensor &output, Tensor &input)
{
    unsigned int step = input.size()/3;


    #ifdef NETWORK_USE_CUDA

        unsigned int size = output.size();

        dim3 block(16);
        dim3 grid((size + block.x + 1)/block.x);

        cuda_highway_forward_kernel<<<grid, block>>>(   output.v,

                                                        input.v + 2*step,
                                                        input.v + 1*step,
                                                        input.v + 0*step,

                                                        output.size() );

        cudaDeviceSynchronize();

    #else

        cpu_highway_forward_kernel( output.v,

                                    input.v + 2*step,
                                    input.v + 1*step,
                                    input.v + 0*step,

                                    output.size() );

    #endif


}

void highway_layer_backward(Tensor &error_back, Tensor &input, Tensor &error)
{
    unsigned int step = input.size()/3;


    #ifdef NETWORK_USE_CUDA

        unsigned int size = error.size();

        dim3 block(16);
        dim3 grid((size + block.x + 1)/block.x);

        cuda_highway_backward_kernel<<<grid, block>>>(  error_back.v + 2*step,
                                                        error_back.v + 1*step,
                                                        error_back.v + 0*step,

                                                        input.v + 2*step,
                                                        input.v + 1*step,
                                                        input.v + 0*step,

                                                        error.v,

                                                        error.size() );
        cudaDeviceSynchronize();

    #else

        cpu_highway_backward_kernel(    error_back.v + 2*step,
                                        error_back.v + 1*step,
                                        error_back.v + 0*step,

                                        input.v + 2*step,
                                        input.v + 1*step,
                                        input.v + 0*step,

                                        error.v,

                                        error.size() );

    #endif
}
