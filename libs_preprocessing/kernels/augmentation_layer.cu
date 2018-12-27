#include "augmentation_layer.cuh"



__host__
void cpu_augmentation_layer_kernel(     float *output,
                                        float *input,
                                        unsigned int width,
                                        unsigned int height,
                                        unsigned int depth,
                                        unsigned int augmentation)
{
    for (unsigned int y = 0; y < height; y++)
    for (unsigned int x = 0; x < height; x++)
    {
        unsigned int x_out = x;
        unsigned int y_out = y;

        switch (augmentation)
        {
            case 0:   x_out = x;
                      y_out = y;
                      break;

            case 1:   x_out = width - 1 - x;
                      y_out = y;
                      break;

            case 2:   x_out = x;
                      y_out = height - 1 - y;
                      break;

            case 3:   x_out = width - 1 - x;
                      y_out = height - 1 - y;
                      break;

            case 4:   x_out = y;
                      y_out = x;
                      break;

            case 5:   x_out = y;
                      y_out = width - 1 - x;
                      break;

            case 6:   x_out = height - 1 - y;;
                      y_out = x;
                      break;

            case 7:   x_out = height - 1 - y;
                      y_out = width - 1 - x;
                      break;
        }

        unsigned int idx_input  = y*width + x;
        unsigned int idx_output = y_out*width + x_out;
        unsigned int size = width*height;

        for (unsigned int ch = 0; ch < depth; ch++)
        {
            output[idx_output] = input[idx_input];
            idx_input+= size;
            idx_output+= size;
        }
    }
}


__global__
void cuda_augmentation_layer_kernel(    float *output,
                                        float *input,
                                        unsigned int width,
                                        unsigned int height,
                                        unsigned int depth,
                                        unsigned int augmentation)
{
  unsigned int x  = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int y  = threadIdx.y + blockIdx.y*blockDim.y;

  if (y < height)
  if (x < width)
  {
      unsigned int x_out = x;
      unsigned int y_out = y;

      switch (augmentation)
      {
          case 0:   x_out = x;
                    y_out = y;
                    break;

          case 1:   x_out = width - 1 - x;
                    y_out = y;
                    break;

          case 2:   x_out = x;
                    y_out = height - 1 - y;
                    break;

          case 3:   x_out = width - 1 - x;
                    y_out = height - 1 - y;
                    break;

          case 4:   x_out = y;
                    y_out = x;
                    break;

          case 5:   x_out = y;
                    y_out = width - 1 - x;
                    break;

          case 6:   x_out = height - 1 - y;;
                    y_out = x;
                    break;

          case 7:   x_out = height - 1 - y;
                    y_out = width - 1 - x;
                    break;
      }

      unsigned int idx_input  = y*width + x;
      unsigned int idx_output = y_out*width + x_out;
      unsigned int size = width*height;

      for (unsigned int ch = 0; ch < depth; ch++)
      {
          output[idx_output] = input[idx_input];
          idx_input+= size;
          idx_output+= size;
      }
  }
}



void augmentation_layer(Tensor &output, Tensor &input, unsigned int augmentation)
{
  #ifdef NETWORK_USE_CUDA

    dim3 block(16, 16);
    dim3 grid(
                (input.w() + block.x - 1)/block.x,
                (input.h() + block.y - 1)/block.y
             );

    cuda_augmentation_layer_kernel<<<grid, block>>>(    output.v,
                                                        input.v,
                                                        input.w(),
                                                        input.h(),
                                                        input.d(),
                                                        augmentation);
    cudaDeviceSynchronize();

  #else

    cpu_augmentation_layer_kernel(   output.v,
                                     input.v,
                                     input.w(),
                                     input.h(),
                                     input.d(),
                                     augmentation);

  #endif
}
