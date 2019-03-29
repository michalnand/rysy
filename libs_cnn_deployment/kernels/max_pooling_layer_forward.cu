#include "max_pooling_layer_forward.cuh"


/*
#define MAX_POOLING_VALUE_MIN   ((float)0.0)

template<unsigned int kernel_size>
__global__
void cuda_max_pooling_forward_kernel( float *output,
                                      float *input,
                                      sGeometry output_geometry)
{
    unsigned int x  = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y  = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int ch = threadIdx.z + blockIdx.z*blockDim.z;

    if (ch < output_geometry.d)
    if (y < output_geometry.h)
    if (x < output_geometry.w)
    {
        unsigned int x_ = x*kernel_size;
        unsigned int y_ = y*kernel_size;

        unsigned int input_width  =  output_geometry.w*kernel_size;
        unsigned int input_height =  output_geometry.h*kernel_size;

        unsigned int input_offset = (ch*input_height + y_)*input_width + x_;

        float result = MAX_POOLING_VALUE_MIN;

        if (kernel_size == 2)
        {
            if (input[input_offset + 0] > result)
                result = input[input_offset + 0];
            if (input[input_offset + 1] > result)
                result = input[input_offset + 1];
            if (input[input_offset + 0 + input_width] > result)
                result = input[input_offset + 0 + input_width];
            if (input[input_offset + 1 + input_width] > result)
                result = input[input_offset + 1 + input_width];
        }

        if (kernel_size == 3)
        {
            if (input[input_offset + 0] > result)
                result = input[input_offset + 0];
            if (input[input_offset + 1] > result)
                result = input[input_offset + 1];
            if (input[input_offset + 2] > result)
                result = input[input_offset + 2];

            if (input[input_offset + 0 + input_width] > result)
                result = input[input_offset + 0 + input_width];
            if (input[input_offset + 1 + input_width] > result)
                result = input[input_offset + 1 + input_width];
            if (input[input_offset + 2 + input_width] > result)
                result = input[input_offset + 2 + input_width];


            if (input[input_offset + 0 + 2*input_width] > result)
                result = input[input_offset + 0 + 2*input_width];
            if (input[input_offset + 1 + 2*input_width] > result)
                result = input[input_offset + 1 + 2*input_width];
            if (input[input_offset + 2 + 2*input_width] > result)
                result = input[input_offset + 2 + 2*input_width];
        }

        unsigned int output_idx = (ch* output_geometry.h + y)* output_geometry.w + x;
        output[output_idx]      = result;
    }
}


void max_pooling_layer_forward( float *input,
                                float *output,
                                sGeometry output_geometry)
{
    dim3 block(16, 16, 1);

    dim3 grid(  (output_geometry.w + block.x + 1)/block.x,
                (output_geometry.h + block.y + 1)/block.y,
                (output_geometry.d + block.z + 1)/block.z);

    cuda_max_pooling_forward_kernel<2><<<grid, block>>>(    output,
                                                            input,
                                                            output_geometry);

    cudaDeviceSynchronize();
}
*/





__global__
void cuda_max_pooling_forward_kernel( float *output,
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
    float result = -100.0;

    unsigned int x_ = x*kernel_width;
    unsigned int y_ = y*kernel_height;

    unsigned int input_width  = output_width*kernel_width;
    unsigned int input_height = output_height*kernel_height;

    for (unsigned int ky = 0; ky < kernel_height; ky++)
      for (unsigned int kx = 0; kx < kernel_width; kx++)
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

void max_pooling_layer_forward( float *input,
                                float *output,
                                sGeometry output_geometry)
{
  unsigned int kernel_width   = 2;
  unsigned int kernel_height  = 2;


      dim3 block(16, 16, 1);
      dim3 grid( (output_geometry.w+block.x-1)/block.x,
                 (output_geometry.h+block.y-1)/block.y,
                 (output_geometry.d+block.z-1)/block.z);

      cuda_max_pooling_forward_kernel<<<grid, block>>>( output,
                                                        input,

                                                        kernel_width,
                                                        kernel_height,

                                                        output_geometry.w,
                                                        output_geometry.h,
                                                        output_geometry.d );

      cudaDeviceSynchronize();

}
