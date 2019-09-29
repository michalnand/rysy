#include "yuv_to_rgb_layer.cuh"



__host__
void cpu_white_noise_layer_kernel(float *output, float *input, float *noise, unsigned int size, float noise_level)
{
  for (unsigned int idx = 0; idx < size; idx++)
  {
    output[idx] = (1.0 - noise_level)*input[idx] + noise_level*noise[idx];
  }
}

__global__
void cuda_white_noise_layer_kernel(float *output, float *input, float *noise, unsigned int size, float noise_level)
{
  unsigned int idx  = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    output[idx] = (1.0 - noise_level)*input[idx] + noise_level*noise[idx];
  }
}



void white_noise_layer(Tensor &output, Tensor &input, Tensor &noise, float noise_level) 
{
  unsigned int size = input.size();

  #ifdef NETWORK_USE_CUDA

    dim3 block(32);
    dim3 grid((size + block.x - 1)/block.x);

    cuda_white_noise_layer_kernel<<<grid, block>>>(output.v, input.v, noise.v, size, noise_level);
    cudaDeviceSynchronize();

  #else

    cpu_white_noise_layer_kernel(output.v, input.v, noise.v, size, noise_level);

  #endif
}
