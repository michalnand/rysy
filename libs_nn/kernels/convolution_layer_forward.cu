#include "convolution_layer_forward.cuh"

#define TILE_MAX_SIZE ((unsigned int)32)

__host__
void cpu_convolution_forward_kernel(   float *output,
                                       float *input,
                                       float *w,
                                       float *bias,

                                       sGeometry output_geometry,
                                       sGeometry input_geometry,
                                       sGeometry kernel_geometry
                                    )
{
  unsigned int kernel_size = kernel_geometry.w;

  unsigned int k_half = (kernel_size - 1)/2;

  unsigned int input_size_y = input_geometry.h - 2*k_half;
  unsigned int input_size_x = input_geometry.w - 2*k_half;

  for (unsigned int filter = 0; filter < kernel_geometry.d; filter++)
    for (unsigned int y = 0; y < input_size_y; y++)
      for (unsigned int x = 0; x < input_size_x; x++)
        {
          unsigned int filter_idx = kernel_geometry.w*kernel_geometry.h*input_geometry.d*filter;

          float sum = 0.0;

          for (unsigned int ch = 0; ch < input_geometry.d; ch++)
            for (unsigned int ky = 0; ky < kernel_geometry.h; ky++)
              for (unsigned int kx = 0; kx < kernel_geometry.w; kx++)
              {
                unsigned int input_idx = (ch*input_geometry.h + y + ky)*input_geometry.w + x + kx;
                sum+= w[filter_idx]*input[input_idx];
                filter_idx++;
              }

          unsigned int output_idx = (filter*input_geometry.h + y + k_half)*input_geometry.w + x + k_half;
          sum+= bias[filter];
          output[output_idx] = sum;
        }
}



template<unsigned int kernel_size>
__global__
void cuda_convolution_forward_kernel(   float *output,
                                        float *input,
                                        float *w,
                                        float *bias,

                                        sGeometry output_geometry,
                                        sGeometry input_geometry,
                                        sGeometry kernel_geometry
                                    )
{
  unsigned int x      = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int y      = threadIdx.y + blockIdx.y*blockDim.y;
  unsigned int z      = threadIdx.z + blockIdx.z*blockDim.z;


  unsigned int ch     = z%input_geometry.d;
  unsigned int filter = z/input_geometry.d;

  unsigned int k_half = (kernel_size - 1)/2;
  unsigned int input_size_y = input_geometry.h - 2*k_half;
  unsigned int input_size_x = input_geometry.w - 2*k_half;

  __shared__ float w_shared[5][5];
  if ( (z < input_geometry.d*output_geometry.d) && (threadIdx.x < kernel_size) && (threadIdx.y < kernel_size) )
  {
    unsigned int w_ofs = z*kernel_size*kernel_size;
    w_shared[threadIdx.y][threadIdx.x] = w[w_ofs + threadIdx.y*kernel_size + threadIdx.x];
  }

  __syncthreads();


  if (filter < output_geometry.d)
  if (ch < input_geometry.d)
  if (y <= input_size_y)
  if (x <= input_size_x)
  {
    unsigned int input_idx  = (ch*input_geometry.h + y)*input_geometry.w + x;
    float sum = 0.0;

    if (kernel_size == 1)
    {
      sum+= w_shared[0][0]*input[input_idx];
      input_idx+= input_geometry.w;
    }


    if (kernel_size == 3)
    {
      sum+= w_shared[0][0]*input[input_idx]; input_idx++;
      sum+= w_shared[0][1]*input[input_idx]; input_idx++;
      sum+= w_shared[0][2]*input[input_idx]; input_idx++;
      input_idx+= input_geometry.w - kernel_size;

      sum+= w_shared[1][0]*input[input_idx]; input_idx++;
      sum+= w_shared[1][1]*input[input_idx]; input_idx++;
      sum+= w_shared[1][2]*input[input_idx]; input_idx++;
      input_idx+= input_geometry.w - kernel_size;

      sum+= w_shared[2][0]*input[input_idx]; input_idx++;
      sum+= w_shared[2][1]*input[input_idx]; input_idx++;
      sum+= w_shared[2][2]*input[input_idx]; input_idx++;
      input_idx+= input_geometry.w - kernel_size;
    }


    if (kernel_size == 5)
    {
      sum+= w_shared[0][0]*input[input_idx]; input_idx++;
      sum+= w_shared[0][1]*input[input_idx]; input_idx++;
      sum+= w_shared[0][2]*input[input_idx]; input_idx++;
      sum+= w_shared[0][3]*input[input_idx]; input_idx++;
      sum+= w_shared[0][4]*input[input_idx]; input_idx++;
      input_idx+= input_geometry.w - kernel_size;

      sum+= w_shared[1][0]*input[input_idx]; input_idx++;
      sum+= w_shared[1][1]*input[input_idx]; input_idx++;
      sum+= w_shared[1][2]*input[input_idx]; input_idx++;
      sum+= w_shared[1][3]*input[input_idx]; input_idx++;
      sum+= w_shared[1][4]*input[input_idx]; input_idx++;
      input_idx+= input_geometry.w - kernel_size;

      sum+= w_shared[2][0]*input[input_idx]; input_idx++;
      sum+= w_shared[2][1]*input[input_idx]; input_idx++;
      sum+= w_shared[2][2]*input[input_idx]; input_idx++;
      sum+= w_shared[2][3]*input[input_idx]; input_idx++;
      sum+= w_shared[2][4]*input[input_idx]; input_idx++;
      input_idx+= input_geometry.w - kernel_size;

      sum+= w_shared[3][0]*input[input_idx]; input_idx++;
      sum+= w_shared[3][1]*input[input_idx]; input_idx++;
      sum+= w_shared[3][2]*input[input_idx]; input_idx++;
      sum+= w_shared[3][3]*input[input_idx]; input_idx++;
      sum+= w_shared[3][4]*input[input_idx]; input_idx++;
      input_idx+= input_geometry.w - kernel_size;

      sum+= w_shared[4][0]*input[input_idx]; input_idx++;
      sum+= w_shared[4][1]*input[input_idx]; input_idx++;
      sum+= w_shared[4][2]*input[input_idx]; input_idx++;
      sum+= w_shared[4][3]*input[input_idx]; input_idx++;
      sum+= w_shared[4][4]*input[input_idx]; input_idx++;
      input_idx+= input_geometry.w - kernel_size;
    }

    unsigned int output_idx = (filter*input_geometry.h + y + k_half)*input_geometry.w + x + k_half;
    sum+= bias[filter]/input_geometry.d;
    atomicAdd(&output[output_idx], sum);
  }
}


__global__
void cuda_convolution_forward_kernel_any_size(  float *output,
                                                float *input,
                                                float *w,
                                                float *bias,

                                                sGeometry output_geometry,
                                                sGeometry input_geometry,
                                                sGeometry kernel_geometry )
{
  unsigned int x      = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int y      = threadIdx.y + blockIdx.y*blockDim.y;
  unsigned int z      = threadIdx.z + blockIdx.z*blockDim.z;

  unsigned int ch     = z%input_geometry.d;
  unsigned int filter = z/input_geometry.d;

  unsigned int k_half_h = (kernel_geometry.h - 1)/2;
  unsigned int k_half_w = (kernel_geometry.w - 1)/2;

  unsigned int input_size_y = input_geometry.h - 2*k_half_h;
  unsigned int input_size_x = input_geometry.w - 2*k_half_w;

  if (filter < output_geometry.d)
  if (ch     < input_geometry.d)
  if (y <= input_size_y)
  if (x <= input_size_x)
  {
    float sum = 0.0;

    unsigned int filter_idx = (filter*input_geometry.d + ch)*kernel_geometry.h*kernel_geometry.w;
    unsigned int input_idx  = (ch*input_geometry.h + y)*input_geometry.w + x;

      for (unsigned int ky = 0; ky < kernel_geometry.h; ky++)
      {
        for (unsigned int kx = 0; kx < kernel_geometry.w; kx++)
        {
          unsigned int input_idx = (ch*input_geometry.h + y + ky)*input_geometry.w + x + kx;
          sum+= w[filter_idx]*input[input_idx];
          filter_idx++;
          input_idx++;
        }

        input_idx+= input_geometry.w - kernel_geometry.w;
      }

    unsigned int output_idx = (filter*input_geometry.h + y + k_half_h)*input_geometry.w + x + k_half_w;
    sum+= bias[filter]/input_geometry.d;
    atomicAdd(&output[output_idx], sum);
  }
}


void convolution_layer_forward(   Tensor &output, Tensor &input,
                                  Tensor &w, Tensor &bias)
{
      unsigned int input_size_y = input.h() - w.h();
      unsigned int input_size_x = input.w() - w.w();

      sGeometry input_geometry;

      input_geometry.w = input.w();
      input_geometry.h = input.h();
      input_geometry.d = input.d();

      sGeometry output_geometry;

      output_geometry.w = output.w();
      output_geometry.h = output.h();
      output_geometry.d = output.d();

      sGeometry kernel_geometry;

      kernel_geometry.w = w.w();
      kernel_geometry.h = w.h();
      kernel_geometry.d = output_geometry.d;


    #ifdef NETWORK_USE_CUDA
      output.clear();


        dim3 block(16, 16, 1);
        dim3 grid( (input_size_x      + block.x - 1)/block.x,
                   (input_size_y      + block.y - 1)/block.y,
                   (kernel_geometry.d*input_geometry.d + block.z - 1)/block.z );

        unsigned int kernel_size = kernel_geometry.w;
        switch (kernel_size)
        {
                  case 1:  cuda_convolution_forward_kernel<1><<<grid, block>>>( output.v,
                                                                                input.v,
                                                                                w.v,
                                                                                bias.v,

                                                                                output_geometry,
                                                                                input_geometry,
                                                                                kernel_geometry);
                              break;

                  case 3:  cuda_convolution_forward_kernel<3><<<grid, block>>>( output.v,
                                                                                input.v,
                                                                                w.v,
                                                                                bias.v,

                                                                                output_geometry,
                                                                                input_geometry,
                                                                                kernel_geometry);
                              break;

                  case 5:  cuda_convolution_forward_kernel<5><<<grid, block>>>( output.v,
                                                                                input.v,
                                                                                w.v,
                                                                                bias.v,

                                                                                output_geometry,
                                                                                input_geometry,
                                                                                kernel_geometry);
                              break;


                  default:
                          cuda_convolution_forward_kernel_any_size<<<grid, block>>>(
                                                                                output.v,
                                                                                input.v,
                                                                                w.v,
                                                                                bias.v,

                                                                                output_geometry,
                                                                                input_geometry,
                                                                                kernel_geometry);
                            break;
          }

          cudaDeviceSynchronize();

    #else

      cpu_convolution_forward_kernel( output.v,
                                      input.v,
                                      w.v,
                                      bias.v,

                                      output_geometry,
                                      input_geometry,
                                      kernel_geometry);
    #endif
}
