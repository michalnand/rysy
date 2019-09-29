#include "max_pooling_layer_forward.cuh"



template<unsigned int kernel_size>
__global__
void cuda_max_pooling_forward_kernel( float *output,
                                      float *input,

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
    float result = -100.0;

    unsigned int x_ = x*kernel_size;
    unsigned int y_ = y*kernel_size;

    unsigned int input_width  = output_width*kernel_size;
    unsigned int input_height = output_height*kernel_size;

    for (unsigned int ky = 0; ky < kernel_size; ky++)
      for (unsigned int kx = 0; kx < kernel_size; kx++)
      {
        unsigned int input_idx = (ch*input_height + y_ + ky)*input_width + x_ + kx;
        float tmp = input[input_idx];
        if (tmp > result)
        {
          result    = tmp;
        }
      }

    unsigned int output_idx = (ch*output_height + y)*output_width + x;
    output[output_idx]      = result;
  }
}

void max_pooling_layer_forward( float *output,
                                float *input,
                                sGeometry output_geometry)
{
      dim3 block(32, 32, 1);
      dim3 grid( (output_geometry.w+block.x-1)/block.x,
                 (output_geometry.h+block.y-1)/block.y,
                 (output_geometry.d+block.z-1)/block.z);

      cuda_max_pooling_forward_kernel<2><<<grid, block>>>(  output,
                                                            input,

                                                            output_geometry.w,
                                                            output_geometry.h,
                                                            output_geometry.d );

      cudaDeviceSynchronize();

}
