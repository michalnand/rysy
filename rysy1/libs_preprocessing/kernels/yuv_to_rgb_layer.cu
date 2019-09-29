#include "yuv_to_rgb_layer.cuh"



__host__
void cpu_yuv_to_rgb_layer_kernel(float *output, float *input, unsigned int width, unsigned int height)
{
  for (unsigned int y = 0; y < height; y++)
  for (unsigned int x = 0; x < width; x++)
  {
    float Y = input[(0*height + y)*width + x];
    float U = input[(1*height + y)*width + x];
    float V = input[(2*height + y)*width + x];

    float r =  1.000*Y + 0.000*U   + 1.13983*V;
    float g =  1.000*Y - 0.39465*U - 0.58060*V;
    float b =  1.000*Y + 2.03211*U + 0.000*V;

    output[(0*height + y)*width + x] = r;
    output[(1*height + y)*width + x] = g;
    output[(2*height + y)*width + x] = b;
  }
}

__global__
void cuda_yuv_to_rgb_layer_kernel(float *output, float *input, unsigned int width, unsigned int height)
{
  unsigned int x  = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int y  = threadIdx.y + blockIdx.y*blockDim.y;

  if (y < height)
  if (x < width)
  {
    float Y = input[(0*height + y)*width + x];
    float U = input[(1*height + y)*width + x];
    float V = input[(2*height + y)*width + x];

    float r =  1.000*Y + 0.000*U   + 1.13983*V;
    float g =  1.000*Y - 0.39465*U - 0.58060*V;
    float b =  1.000*Y + 2.03211*U + 0.000*V;

    output[(0*height + y)*width + x] = r;
    output[(1*height + y)*width + x] = g;
    output[(2*height + y)*width + x] = b;
  }
}



void yuv_to_rgb_layer(Tensor &output, Tensor &input)
{
  #ifdef NETWORK_USE_CUDA

    dim3 block(16, 16);
    dim3 grid(  (input.w() + block.x - 1)/block.x,
                (input.h() + block.y - 1)/block.y );

    cuda_yuv_to_rgb_layer_kernel<<<grid, block>>>(output.v, input.v, input.w(), input.h());
    cudaDeviceSynchronize();

  #else

    cpu_yuv_to_rgb_layer_kernel(output.v, input.v, input.w(), input.h());

  #endif
}
