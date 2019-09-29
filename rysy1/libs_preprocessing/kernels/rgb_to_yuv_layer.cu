#include "rgb_to_yuv_layer.cuh"



__host__
void cpu_rgb_to_yuv_layer_kernel(float *output, float *input, unsigned int width, unsigned int height)
{
  for (unsigned int y = 0; y < height; y++)
  for (unsigned int x = 0; x < width; x++)
  {
    float r = input[(0*height + y)*width + x];
    float g = input[(1*height + y)*width + x];
    float b = input[(2*height + y)*width + x];

    float Y =  0.299*r + 0.587*g + 0.114*b;
    float U = -0.147*r - 0.289*g + 0.436*b;
    float V =  0.615*r - 0.515*g - 0.100*b;

    output[(0*height + y)*width + x] = Y;
    output[(1*height + y)*width + x] = U;
    output[(2*height + y)*width + x] = V;
  }
}

__global__
void cuda_rgb_to_yuv_layer_kernel(float *output, float *input, unsigned int width, unsigned int height)
{
  unsigned int x  = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int y  = threadIdx.y + blockIdx.y*blockDim.y;

  if (y < height)
  if (x < width)
  {
    float r = input[(0*height + y)*width + x];
    float g = input[(1*height + y)*width + x];
    float b = input[(2*height + y)*width + x];

    float Y =  0.299*r + 0.587*g + 0.114*b;
    float U = -0.147*r - 0.289*g + 0.436*b;
    float V =  0.615*r - 0.515*g - 0.100*b;

    output[(0*height + y)*width + x] = Y;
    output[(1*height + y)*width + x] = U;
    output[(2*height + y)*width + x] = V;
  }
}



void rgb_to_yuv_layer(Tensor &output, Tensor &input)
{
  #ifdef NETWORK_USE_CUDA

    dim3 block(16, 16);
    dim3 grid(  (input.w() + block.x - 1)/block.x,
                (input.h() + block.y - 1)/block.y );

    cuda_rgb_to_yuv_layer_kernel<<<grid, block>>>(output.v, input.v, input.w(), input.h());
    cudaDeviceSynchronize();

  #else

    cpu_rgb_to_yuv_layer_kernel(output.v, input.v, input.w(), input.h());

  #endif
}
