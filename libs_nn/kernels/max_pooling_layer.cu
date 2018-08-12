#include "max_pooling_layer.cuh"

#define MAX_POOLING_VALUE_MIN   ((float)-10000000.0)

__host__
void cpu_max_pooling_forward_kernel(  float *output,
                                      float *input,
                                      float *max_mask,

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
    float result = MAX_POOLING_VALUE_MIN;

    unsigned int x_ = x*kernel_width;
    unsigned int y_ = y*kernel_height;

    unsigned int input_width  = output_width*kernel_width;
    unsigned int input_height = output_height*kernel_height;

    unsigned int mask_idx     = 0;

    for (unsigned int ky = 0; ky < kernel_height; ky++)
      for (unsigned int kx = 0; kx < kernel_width; kx++)
      {
        unsigned int input_idx = (ch*input_height + y_ + ky)*input_width + x_ + kx;
        float tmp = input[input_idx];
        if (tmp > result)
        {
          result    = tmp;
          mask_idx  = input_idx;
        }
      }

    unsigned int output_idx = (ch*output_height + y)*output_width + x;
    output[output_idx]      = result;
    max_mask[mask_idx]      = 1.0;
  }
}

__global__
void cuda_max_pooling_forward_kernel( float *output,
                                      float *input,
                                      float *max_mask,

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
    float result = MAX_POOLING_VALUE_MIN;

    unsigned int x_ = x*kernel_width;
    unsigned int y_ = y*kernel_height;

    unsigned int input_width  = output_width*kernel_width;
    unsigned int input_height = output_height*kernel_height;

    unsigned int mask_idx = 0;

    for (unsigned int ky = 0; ky < kernel_height; ky++)
      for (unsigned int kx = 0; kx < kernel_width; kx++)
      {
        unsigned int input_idx = (ch*input_height + y_ + ky)*input_width + x_ + kx;
        float tmp = input[input_idx];
        if (tmp > result)
        {
          result    = tmp;
          mask_idx  = input_idx;
        }
      }

    unsigned int output_idx = (ch*output_height + y)*output_width + x;
    output[output_idx]      = result;
    max_mask[mask_idx]      = 1.0;
  }
}

void max_pooling_layer_forward(Tensor &max_mask, Tensor &output, Tensor &input)
{
  max_mask.clear();

  unsigned int kernel_width   = input.w()/output.w();
  unsigned int kernel_height  = input.h()/output.h();

  #ifdef NETWORK_USE_CUDA

      dim3 block(16, 16, 1);
      dim3 grid( (output.w()+block.x-1)/block.x,
                 (output.h()+block.y-1)/block.y,
                 (output.d()+block.z-1)/block.z);

      cuda_max_pooling_forward_kernel<<<grid, block>>>( output.v,
                                                        input.v,
                                                        max_mask.v,

                                                        kernel_width,
                                                        kernel_height,

                                                        output.w(),
                                                        output.h(),
                                                        output.d() );

      cudaDeviceSynchronize();

  #else

      cpu_max_pooling_forward_kernel( output.v,
                                      input.v,
                                      max_mask.v,

                                      kernel_width,
                                      kernel_height,

                                      output.w(),
                                      output.h(),
                                      output.d() );

  #endif
}









__host__
void cpu_max_pooling_backward_kernel(  float *error_back,
                                       float *error,
                                       float *max_mask,

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
    float result = error[error_idx];

    unsigned int error_back_height  = error_height*kernel_height;
    unsigned int error_back_width   = error_width*kernel_width;

    for (unsigned int ky = 0; ky < kernel_height; ky++)
      for (unsigned int kx = 0; kx < kernel_width; kx++)
      {
        unsigned int error_back_idx = (ch*error_back_height + y*kernel_height + ky)*error_back_width + x*kernel_width + kx;
        error_back[error_back_idx] = result*max_mask[error_back_idx];
      }
  }
}

__global__
void cuda_max_pooling_backward_kernel( float *error_back,
                                       float *error,
                                       float *max_mask,

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
    float result = error[error_idx];

    unsigned int error_back_height  = error_height*kernel_height;
    unsigned int error_back_width   = error_width*kernel_width;

    for (unsigned int ky = 0; ky < kernel_height; ky++)
      for (unsigned int kx = 0; kx < kernel_width; kx++)
      {
        unsigned int error_back_idx = (ch*error_back_height + y*kernel_height + ky)*error_back_width + x*kernel_width + kx;
        error_back[error_back_idx] = result*max_mask[error_back_idx];
      }
  }
}


void max_pooling_layer_backward(Tensor &error_back, Tensor &error, Tensor &max_mask)
{
  error_back.clear();

  unsigned int kernel_width   = error_back.w()/error.w();
  unsigned int kernel_height  = error_back.h()/error.h();

  #ifdef NETWORK_USE_CUDA

    dim3 block(16, 16, 1);
    dim3 grid(  (error.w()+block.x-1)/block.x,
                (error.h()+block.y-1)/block.y,
                (error.d()+block.z-1)/block.z);


    cuda_max_pooling_backward_kernel<<<grid, block>>>(  error_back.v,
                                                        error.v,
                                                        max_mask.v,

                                                        kernel_width,
                                                        kernel_height,

                                                        error.w(),
                                                        error.h(),
                                                        error.d() );

    cudaDeviceSynchronize();

  #else


    cpu_max_pooling_backward_kernel(  error_back.v,
                                      error.v,
                                      max_mask.v,

                                      kernel_width,
                                      kernel_height,

                                      error.w(),
                                      error.h(),
                                      error.d() );

  #endif
}
