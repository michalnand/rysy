#include "yuv_to_rgb_layer.cuh"



__host__
void cpu_luma_noise_layer_kernel(float *output, float *input, unsigned int size, float noise)
{
  for (unsigned int idx = 0; idx < size; idx++)
  {
    output[idx] = input[idx] + noise;
  }
}

__global__
void cuda_luma_noise_layer_kernel(float *output, float *input, unsigned int size, float noise)
{
  unsigned int idx  = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    output[idx] = input[idx] + noise;
  }
}



void luma_noise_layer(Tensor &output, Tensor &input, float noise)
{
  unsigned int size = input.size();

  #ifdef NETWORK_USE_CUDA

    dim3 block(32);
    dim3 grid((size + block.x - 1)/block.x);

    cuda_luma_noise_layer_kernel<<<grid, block>>>(output.v, input.v, size, noise);
    cudaDeviceSynchronize();

  #else

    cpu_luma_noise_layer_kernel(output.v, input.v, size, noise);

  #endif
}
