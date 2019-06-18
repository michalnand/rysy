#include "unpooling_layer.cuh"

__host__
void cpu_unpooling_forward_kernel(  float *output,
                                    float *input,

                                    unsigned int kernel_width,
                                    unsigned int kernel_height,

                                    unsigned int input_width,
                                    unsigned int input_height,
                                    unsigned int input_depth)
{
  for (unsigned int ch = 0; ch < input_depth; ch++)
  for (unsigned int y = 0; y < input_height; y++)
  for (unsigned int x = 0; x < input_width; x++)
  {
    unsigned int input_idx = (ch*input_height + y)*input_width + x;

    unsigned int output_height  = input_height*kernel_height;
    unsigned int output_width   = input_width*kernel_width;

    float result = input[input_idx];

    for (unsigned int ky = 0; ky < kernel_height; ky++)
      for (unsigned int kx = 0; kx < kernel_width; kx++)
      {
        unsigned int y_ = y*kernel_width + ky;
        unsigned int x_ = x*kernel_width + kx;

        unsigned int output_idx = (ch*output_height + y_)*output_width + x_;
        output[output_idx] = result;
      }
  }
}

__global__
void cuda_unpooling_forward_kernel( float *output,
                                    float *input,

                                    unsigned int kernel_width,
                                    unsigned int kernel_height,

                                    unsigned int input_width,
                                    unsigned int input_height,
                                    unsigned int input_depth)
{
  unsigned int x  = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int y  = threadIdx.y + blockIdx.y*blockDim.y;
  unsigned int ch = threadIdx.z + blockIdx.z*blockDim.z;

  if (ch < input_depth)
  if (y < input_height)
  if (x < input_width)
  {
    unsigned int input_idx = (ch*input_height + y)*input_width + x;

    unsigned int output_height  = input_height*kernel_height;
    unsigned int output_width   = input_width*kernel_width;

    float result = input[input_idx];

    for (unsigned int ky = 0; ky < kernel_height; ky++)
      for (unsigned int kx = 0; kx < kernel_width; kx++)
      {
        unsigned int y_ = y*kernel_width + ky;
        unsigned int x_ = x*kernel_width + kx;

        unsigned int output_idx = (ch*output_height + y_)*output_width + x_;
        output[output_idx] = result;
      }
  }
}

void unpooling_layer_forward(Tensor &output, Tensor &input)
{
  unsigned int kernel_width   = output.w()/input.w();
  unsigned int kernel_height  = output.h()/input.h();

  #ifdef NETWORK_USE_CUDA

      dim3 block(16, 16, 1);
      dim3 grid( (input.w()+block.x+1)/block.x,
                 (input.h()+block.y+1)/block.y,
                 (input.d()+block.z+1)/block.z);

      cuda_unpooling_forward_kernel<<<grid, block>>>( output.v,
                                                      input.v,

                                                      kernel_width,
                                                      kernel_height,

                                                      input.w(),
                                                      input.h(),
                                                      input.d() );

      cudaDeviceSynchronize();

  #else

      cpu_unpooling_forward_kernel( output.v,
                                    input.v,

                                    kernel_width,
                                    kernel_height,

                                    input.w(),
                                    input.h(),
                                    input.d() );

  #endif
}









__host__
void cpu_unpooling_backward_kernel( float *error_back,
                                    float *error,

                                    unsigned int kernel_width,
                                    unsigned int kernel_height,

                                    unsigned int error_back_width,
                                    unsigned int error_back_height,
                                    unsigned int error_back_depth)
{
  for (unsigned int ch = 0; ch <  error_back_depth; ch++)
  for (unsigned int y  = 0; y <   error_back_height; y++)
  for (unsigned int x  = 0; x <   error_back_width; x++)
  {
    unsigned int error_height   = error_back_height*kernel_height;
    unsigned int error_width    = error_back_width*kernel_width;
    unsigned int error_back_idx = (ch*error_back_height + y)*error_back_width + x;

    float result = 0.0;

    for (unsigned int ky = 0; ky < kernel_height; ky++)
      for (unsigned int kx = 0; kx < kernel_width; kx++)
      {
        unsigned int error_idx = (ch*error_height + y*kernel_height + ky)*error_width + x*kernel_width + kx;
        result+= error[error_idx];
      }

    error_back[error_back_idx] = result/(kernel_height*kernel_width);
  }
}

__global__
void cuda_unpooling_backward_kernel( float *error_back,
                                     float *error,

                                     unsigned int kernel_width,
                                     unsigned int kernel_height,

                                     unsigned int error_back_width,
                                     unsigned int error_back_height,
                                     unsigned int error_back_depth)
{
  unsigned int x  = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int y  = threadIdx.y + blockIdx.y*blockDim.y;
  unsigned int ch = threadIdx.z + blockIdx.z*blockDim.z;

  if (ch < error_back_depth)
  if (y < error_back_height)
  if (x < error_back_width)
  {
    unsigned int error_height   = error_back_height*kernel_height;
    unsigned int error_width    = error_back_width*kernel_width;
    unsigned int error_back_idx = (ch*error_back_height + y)*error_back_width + x;

    float result = 0.0;

    for (unsigned int ky = 0; ky < kernel_height; ky++)
      for (unsigned int kx = 0; kx < kernel_width; kx++)
      {
        unsigned int error_idx = (ch*error_height + y*kernel_height + ky)*error_width + x*kernel_width + kx;
        result+= error[error_idx];
      }

    error_back[error_back_idx] = result/(kernel_height*kernel_width);
  }
}


void unpooling_layer_backward(Tensor &error_back, Tensor &error)
{
  error_back.clear();

  unsigned int kernel_width   = error.w()/error_back.w();
  unsigned int kernel_height  = error.h()/error_back.h();

  #ifdef NETWORK_USE_CUDA

    dim3 block(16, 16, 1);
    dim3 grid(  (error_back.w()+block.x+1)/block.x,
                (error_back.h()+block.y+1)/block.y,
                (error_back.d()+block.z+1)/block.z);


    cuda_unpooling_backward_kernel<<<grid, block>>>(  error_back.v,
                                                      error.v,

                                                      kernel_width,
                                                      kernel_height,

                                                      error_back.w(),
                                                      error_back.h(),
                                                      error_back.d() );

    cudaDeviceSynchronize();

  #else


    cpu_unpooling_backward_kernel(  error_back.v,
                                    error.v,

                                    kernel_width,
                                    kernel_height,

                                    error_back.w(),
                                    error_back.h(),
                                    error_back.d() );

  #endif
}
