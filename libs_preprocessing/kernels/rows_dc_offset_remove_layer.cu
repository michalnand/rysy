#include "rows_dc_offset_remove_layer.cuh"



__host__
void cpu_rows_dc_offset_remove_layer_kernel( float *output, float *input,
                                        unsigned int width, unsigned height, unsigned int depth)
{
  for (unsigned int channel = 0; channel < depth; channel++)
  for (unsigned int row = 0; row < height; row++)
  for (unsigned int column = 0; column < (width-1); column++)
  {
    unsigned int idx = (channel*height + row)*width + column;

    output[idx] = input[idx] - input[idx+1];
  }
}

__global__
void cuda_rows_dc_offset_remove_layer_kernel( float *output, float *input,
                                        unsigned int width, unsigned int height, unsigned int depth)
{
  unsigned int column     = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int row        = threadIdx.y + blockIdx.y*blockDim.y;
  unsigned int channel    = threadIdx.z + blockIdx.z*blockDim.z;

  if (channel < depth)
  if (row < height)
  if (column < (width-1))
  {
    unsigned int idx = (channel*height + row)*width + column;

    output[idx] = input[idx] - input[idx+1];
  }
}



void rows_dc_offset_remove_layer(Tensor &output, Tensor &input)
{
  output.clear();
  
  #ifdef NETWORK_USE_CUDA

    dim3 block(8, 8, 1);
    dim3 grid((input.w() + block.x - 1)/block.x,
              (input.h() + block.y - 1)/block.y,
              (input.d() + block.z - 1)/block.z);

    cuda_rows_dc_offset_remove_layer_kernel<<<grid, block>>>(output.v, input.v, input.w(), input.h(), input.d());
    cudaDeviceSynchronize();
  #else

    cpu_rows_dc_offset_remove_layer_kernel(output.v, input.v, input.w(), input.h(), input.d());

  #endif
}
