#include "crop_layer.cuh"

 
__host__
void cpu_crop_forward_kernel(   float *output, float *input,
                                unsigned int out_width, unsigned int out_height, unsigned int out_depth,
                                unsigned int k_width, unsigned int k_height)
{
  for (unsigned int z = 0; z < out_depth; z++)
  for (unsigned int y = 0; y < out_height; y++)
  {
    unsigned int input_height = out_height + 2*k_height;
    unsigned int input_width  = out_width  + 2*k_width;

    unsigned int output_idx = (z*out_height + y)*out_width;
    unsigned int input_idx  = (z*input_height + y + k_height)*input_width + k_width;

    for (unsigned int i = 0; i < out_width; i++)
    {
      output[output_idx] = input[input_idx];
      output_idx++;
      input_idx++;
    }
  }
}

__global__
void cuda_crop_forward_kernel(  float *output, float *input,
                                unsigned int out_width, unsigned int out_height, unsigned int out_depth,
                                unsigned int k_width, unsigned int k_height)
{
  unsigned int y = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int z = threadIdx.y + blockIdx.y*blockDim.y;

  if (z < out_depth)
  if (y < out_height)
  {
    unsigned int input_height = out_height + 2*k_height;
    unsigned int input_width  = out_width  + 2*k_width;

    unsigned int output_idx = (z*out_height + y)*out_width;
    unsigned int input_idx  = (z*input_height + y + k_height)*input_width + k_width;

    for (unsigned int i = 0; i < out_width; i++)
    {
      output[output_idx] = input[input_idx];
      output_idx++;
      input_idx++;
    }
  }
}

void crop_layer_forward(Tensor &output, Tensor &input)
{
  #ifdef NETWORK_USE_CUDA
    dim3 block(8, 8);
    dim3 grid((output.h()      + block.x - 1)/block.x,
              (output.d()      + block.y - 1)/block.y );

    cuda_crop_forward_kernel<<<grid, block>>>(  output.v, input.v,
                                                output.w(), output.h(), output.d(),
                                                input.w() - output.w(),
                                                input.h() - output.h() );

    cudaDeviceSynchronize();

  #else

    cpu_crop_forward_kernel(  output.v, input.v,
                              output.w(), output.h(), output.d(),
                              input.w() - output.w(),
                              input.h() - output.h() );

  #endif

}


__host__
void cpu_crop_backward_kernel(  float *error_back, float *error,
                                unsigned int out_width, unsigned int out_height, unsigned int out_depth,
                                unsigned int k_width, unsigned int k_height)
{
  for (unsigned int z = 0; z < out_depth; z++)
  for (unsigned int y = 0; y < out_height; y++)
  {
    unsigned int input_height = out_height + 2*k_height;
    unsigned int input_width  = out_width  + 2*k_width;

    unsigned int output_idx = (z*out_height + y)*out_width;
    unsigned int input_idx  = (z*input_height + y + k_height)*input_width + k_width;

    for (unsigned int i = 0; i < out_width; i++)
    {
      error_back[input_idx] = error[output_idx];

      output_idx++;
      input_idx++;
    }
  }
}

__global__
void cuda_crop_backward_kernel(  float *error_back, float *error,
                                 unsigned int out_width, unsigned int out_height, unsigned int out_depth,
                                 unsigned int k_width, unsigned int k_height)
{
  unsigned int y = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int z = threadIdx.y + blockIdx.y*blockDim.y;

  if (z < out_depth)
  if (y < out_height)
  {
    unsigned int input_height = out_height + 2*k_height;
    unsigned int input_width  = out_width  + 2*k_width;

    unsigned int output_idx = (z*out_height + y)*out_width;
    unsigned int input_idx  = (z*input_height + y + k_height)*input_width + k_width;

    for (unsigned int i = 0; i < out_width; i++)
    {
      error_back[input_idx] = error[output_idx];

      output_idx++;
      input_idx++;
    }
  }
}

void crop_layer_backward(Tensor &error_back, Tensor &error)
{
  error_back.clear();

  #ifdef NETWORK_USE_CUDA

    dim3 block(8, 8);
    dim3 grid((error.h()      + block.x - 1)/block.x,
              (error.d()      + block.y - 1)/block.y );

    cuda_crop_backward_kernel<<<grid, block>>>( error_back.v, error.v,
                                                error.w(), error.h(), error.d(),
                                                error_back.w() - error.w(),
                                                error_back.h() - error.h() );

    cudaDeviceSynchronize();

  #else

    cpu_crop_backward_kernel( error_back.v, error.v,
                              error.w(), error.h(), error.d(),
                              error_back.w() - error.w(),
                              error_back.h() - error.h() );

  #endif
}
