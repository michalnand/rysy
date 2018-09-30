#include "noise_layer.cuh"

__host__
void cpu_noise_forward_kernel(
                                float *output, float *input, unsigned int size,

                                float *white_noise,
                                float *salt_and_pepper_noise,
                                float brightness_noise,

                                float white_level,
                                float salt_and_pepper_level,
                                float brightness_level
                              )
{
  for (unsigned int idx = 0; idx < size; idx++)
  {
    float result = (1.0 - white_level)*input[idx] + white_level*white_noise[idx]; // + brightness_noise*brightness_level;

    /*
    if (result < 0.0)
      result = 0.0;
    if (result > 1.0)
      result = 1.0;
    */ 
    /*
    if (salt_and_pepper_noise[idx] > salt_and_pepper_level)
      result = 1.0;

    if (salt_and_pepper_noise[idx] < -salt_and_pepper_level)
      result = 0.0;
    */
    output[idx] = result;
  }
}


__global__
void cuda_noise_forward_kernel(
                                float *output, float *input, unsigned int size,

                                float *white_noise,
                                float white_level,

                                float brightness_noise
                              )
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    float result = (1.0 - white_level)*input[idx] + white_level*white_noise[idx] + brightness_noise;

    /*
    if (salt_and_pepper_noise[idx] > salt_and_pepper_level)
      result = 1.0;

    if (salt_and_pepper_noise[idx] < -salt_and_pepper_level)
      result = 0.0;
    */

    if (result < 0.0)
      result = 0.0;
    if (result > 1.0)
      result = 1.0;
    output[idx] = result;
  }
}

void noise_layer_forward(
                          Tensor &output, Tensor &input, sHyperparameters hyperparameters,
                          Tensor &white_noise, Tensor &salt_and_pepper_noise, float brightness_noise
                        )
{
  unsigned int size = output.size();

  float white_level, salt_and_pepper_level, brightness_level;

  white_level           = hyperparameters.noise;
  salt_and_pepper_level = 0.1*hyperparameters.noise;
  brightness_level      = tanh(2.0*hyperparameters.noise);


  #ifdef NETWORK_USE_CUDA

    unsigned int block_size = 16;
    if (size > 256)
      block_size = 256;

    dim3 block(block_size);
    dim3 grid((size + block.x - 1)/block.x);

    cuda_noise_forward_kernel<<<grid, block>>>(
                                                output.v, input.v, size,
                                                white_noise.v,
                                                white_level,
                                                brightness_noise*brightness_level
                                              );
    cudaDeviceSynchronize();

  #else

    cpu_noise_forward_kernel(
                              output.v, input.v, size,
                              white_noise.v, salt_and_pepper_noise.v, brightness_noise,

                              white_level, salt_and_pepper_level, brightness_level
                            );

  #endif
}
