#include "average_pooling_layer.cuh"


__host__
void cpu_average_pooling_forward_kernel(    float *output,
                                            float *input,

                                            unsigned int kernel_width,
                                            unsigned int kernel_height,

                                            unsigned int output_width,
                                            unsigned int output_height,
                                            unsigned int output_depth)
{
    for (unsigned int ch = 0; ch < output_depth; ch++)
    for (unsigned int y = 0; y < output_height; y++)
    for (unsigned int x = 0; x < output_width; x++)
    {
        float result = 0.0;

        unsigned int x_ = x*kernel_width;
        unsigned int y_ = y*kernel_height;

        unsigned int input_width  = output_width*kernel_width;
        unsigned int input_height = output_height*kernel_height;

        for (unsigned int ky = 0; ky < kernel_height; ky++)
        {
            unsigned int input_idx = (ch*input_height + y_ + ky)*input_width + x_;
            for (unsigned int kx = 0; kx < kernel_width; kx++)
            {
                result+= input[input_idx];
                input_idx++;
            }
        }

        unsigned int output_idx = (ch*output_height + y)*output_width + x;
        output[output_idx]      = result/(kernel_width*kernel_height);
    }
}

__global__
void cuda_average_pooling_forward_kernel( float *output,
                                      float *input,

                                      unsigned int kernel_width,
                                      unsigned int kernel_height,

                                      unsigned int output_width,
                                      unsigned int output_height,
                                      unsigned int output_depth)
{
    unsigned int x  = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y  = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int ch = threadIdx.z + blockIdx.z*blockDim.z;

    if (ch < output_depth)
    if (y < output_height)
    if (x < output_width)
    {
        float result = 0.0;

        unsigned int x_ = x*kernel_width;
        unsigned int y_ = y*kernel_height;

        unsigned int input_width  = output_width*kernel_width;
        unsigned int input_height = output_height*kernel_height;

        for (unsigned int ky = 0; ky < kernel_height; ky++)
        {
            unsigned int input_idx = (ch*input_height + y_ + ky)*input_width + x_;
            for (unsigned int kx = 0; kx < kernel_width; kx++)
            {
                result+= input[input_idx];
                input_idx++;
            }
        }

        unsigned int output_idx = (ch*output_height + y)*output_width + x;
        output[output_idx]      = result/(kernel_width*kernel_height);
    }
}

void average_pooling_layer_forward(Tensor &output, Tensor &input)
{
    unsigned int kernel_width   = input.w()/output.w();
    unsigned int kernel_height  = input.h()/output.h();

    #ifdef NETWORK_USE_CUDA

      dim3 block(2, 2, 16);
      dim3 grid( (output.w() + block.x + 1)/block.x,
                 (output.h() + block.y + 1)/block.y,
                 (output.d() + block.z + 1)/block.z);

      cuda_average_pooling_forward_kernel<<<grid, block>>>( output.v,
                                                            input.v,

                                                            kernel_width,
                                                            kernel_height,

                                                            output.w(),
                                                            output.h(),
                                                            output.d() );

      cudaDeviceSynchronize();

    #else

      cpu_average_pooling_forward_kernel(   output.v,
                                            input.v,

                                            kernel_width,
                                            kernel_height,

                                            output.w(),
                                            output.h(),
                                            output.d() );

  #endif
}









__host__
void cpu_average_pooling_backward_kernel(   float *error_back,
                                            float *error,

                                            unsigned int kernel_width,
                                            unsigned int kernel_height,

                                            unsigned int error_width,
                                            unsigned int error_height,
                                            unsigned int error_depth)
{
    for (unsigned int ch = 0; ch < error_depth; ch++)
    for (unsigned int y  = 0; y < error_height; y++)
    for (unsigned int x  = 0; x < error_width; x++)
    {
        unsigned int error_idx = (ch*error_height + y)*error_width + x;
        float result = error[error_idx]/(kernel_width*kernel_height);

        unsigned int error_back_height  = error_height*kernel_height;
        unsigned int error_back_width   = error_width*kernel_width;

        for (unsigned int ky = 0; ky < kernel_height; ky++)
        {
            unsigned int error_back_idx = (ch*error_back_height + y*kernel_height + ky)*error_back_width + x*kernel_width;
            for (unsigned int kx = 0; kx < kernel_width; kx++)
            {
                error_back[error_back_idx] = result;
                error_back_idx++;
            }
        }
    }
}

__global__
void cuda_average_pooling_backward_kernel(  float *error_back,
                                            float *error,

                                            unsigned int kernel_width,
                                            unsigned int kernel_height,

                                            unsigned int error_width,
                                            unsigned int error_height,
                                            unsigned int error_depth)
{
    unsigned int x  = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y  = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int ch = threadIdx.z + blockIdx.z*blockDim.z;

    if (ch < error_depth)
    if (y < error_height)
    if (x < error_width)
    {
        unsigned int error_idx = (ch*error_height + y)*error_width + x;
        float result = error[error_idx]/(kernel_width*kernel_height);

        unsigned int error_back_height  = error_height*kernel_height;
        unsigned int error_back_width   = error_width*kernel_width;

        for (unsigned int ky = 0; ky < kernel_height; ky++)
        {
            unsigned int error_back_idx = (ch*error_back_height + y*kernel_height + ky)*error_back_width + x*kernel_width;
            for (unsigned int kx = 0; kx < kernel_width; kx++)
            {
                error_back[error_back_idx] = result;
                error_back_idx++;
            }
        }
    }
}


void average_pooling_layer_backward(Tensor &error_back, Tensor &error)
{
  error_back.clear();

  unsigned int kernel_width   = error_back.w()/error.w();
  unsigned int kernel_height  = error_back.h()/error.h();

  #ifdef NETWORK_USE_CUDA

    dim3 block(2, 2, 16);
    dim3 grid(  (error.w()+ block.x + 1)/block.x,
                (error.h()+ block.y + 1)/block.y,
                (error.d()+ block.z + 1)/block.z);


    cuda_average_pooling_backward_kernel<<<grid, block>>>(  error_back.v,
                                                            error.v,

                                                            kernel_width,
                                                            kernel_height,

                                                            error.w(),
                                                            error.h(),
                                                            error.d() );

    cudaDeviceSynchronize();

  #else


    cpu_average_pooling_backward_kernel(    error_back.v,
                                            error.v,

                                            kernel_width,
                                            kernel_height,

                                            error.w(),
                                            error.h(),
                                            error.d() );

  #endif
}
