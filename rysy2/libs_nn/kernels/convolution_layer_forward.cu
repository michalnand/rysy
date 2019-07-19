#include "convolution_layer_forward.cuh"

#define TILE_MAX_SIZE ((unsigned int)32)
#define MAX_FEATURE_MAP_COUNT ((unsigned int)1024)

__host__
void cpu_convolution_forward_kernel(   float *output,
                                       float *input,
                                       float *w,
                                       float *bias,

                                       sShape output_shape,
                                       sShape input_shape,
                                       sShape kernel_shape
                                    )
{
  unsigned int kernel_size = kernel_shape.w;

  unsigned int k_half = (kernel_size - 1)/2;

  unsigned int input_size_y = input_shape.h - 2*k_half;
  unsigned int input_size_x = input_shape.w - 2*k_half;

  for (unsigned int filter = 0; filter < kernel_shape.d; filter++)
    for (unsigned int y = 0; y < input_size_y; y++)
      for (unsigned int x = 0; x < input_size_x; x++)
        {
          unsigned int filter_idx = kernel_shape.w*kernel_shape.h*input_shape.d*filter;

          float sum = 0.0;

          for (unsigned int ch = 0; ch < input_shape.d; ch++)
            for (unsigned int ky = 0; ky < kernel_shape.h; ky++)
              for (unsigned int kx = 0; kx < kernel_shape.w; kx++)
              {
                unsigned int input_idx = (ch*input_shape.h + y + ky)*input_shape.w + x + kx;
                sum+= w[filter_idx]*input[input_idx];
                filter_idx++;
              }

          unsigned int output_idx = (filter*input_shape.h + y + k_half)*input_shape.w + x + k_half;
          sum+= bias[filter];
          output[output_idx] = sum;
        }
}


/*
template<unsigned int kernel_size>
__global__
void cuda_convolution_forward_kernel(   float *output,
                                        float *input,
                                        float *w,
                                        float *bias,

                                        sShape output_shape,
                                        sShape input_shape,
                                        sShape kernel_shape
                                    )
{
    unsigned int x      = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y      = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int z      = threadIdx.z + blockIdx.z*blockDim.z;

    unsigned int filter = z;

    unsigned int k_half = (kernel_size - 1)/2;
    unsigned int input_size_y = input_shape.h - 2*k_half;
    unsigned int input_size_x = input_shape.w - 2*k_half;

    __shared__ float w_shared[MAX_FEATURE_MAP_COUNT][kernel_size][kernel_size];
    if ( (threadIdx.x < kernel_size) && (threadIdx.y < kernel_size) )
    for (unsigned int ch = 0; ch < input_shape.d; ch++)
    {
        unsigned int w_ofs = kernel_size*kernel_size*ch + filter*kernel_size*kernel_size*input_shape.d;
        w_shared[ch][threadIdx.y][threadIdx.x] = w[w_ofs + threadIdx.y*kernel_size + threadIdx.x];
    }

    __syncthreads();


    if (filter < output_shape.d)
    if (y < input_size_y)
    if (x < input_size_x)
    {
        float sum = bias[filter];

        for (unsigned int ch = 0; ch < input_shape.d; ch++)
        {
            unsigned int input_idx  = (ch*input_shape.h + y)*input_shape.w + x;

            if (kernel_size == 1)
            {
                sum+= w_shared[ch][0][0]*input[input_idx];
                input_idx+= input_shape.w;
            }

            if (kernel_size == 3)
            {
                sum+= w_shared[ch][0][0]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][0][1]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][0][2]*input[input_idx]; input_idx++;
                input_idx+= input_shape.w - kernel_size;

                sum+= w_shared[ch][1][0]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][1][1]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][1][2]*input[input_idx]; input_idx++;
                input_idx+= input_shape.w - kernel_size;

                sum+= w_shared[ch][2][0]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][2][1]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][2][2]*input[input_idx]; input_idx++;
                input_idx+= input_shape.w - kernel_size;
            }
        }

        unsigned int output_idx = (filter*input_shape.h + y + k_half)*input_shape.w + x + k_half;
        output[output_idx] = sum;
    }
}
*/



template<unsigned int kernel_size>
__global__
void cuda_convolution_forward_kernel(   float *output,
                                        float *input,
                                        float *w,
                                        float *bias,

                                        sShape output_shape,
                                        sShape input_shape,
                                        sShape kernel_shape
                                    )
{
    unsigned int x      = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y      = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int z      = threadIdx.z + blockIdx.z*blockDim.z;

    unsigned int filter = z;

    unsigned int k_half = (kernel_size - 1)/2;
    unsigned int input_size_y = input_shape.h - 2*k_half;
    unsigned int input_size_x = input_shape.w - 2*k_half;
 
    if (filter < output_shape.d)
    if (y <= input_size_y)
    if (x <= input_size_x)
    {
        float sum = bias[filter];
        __shared__ float w_shared[kernel_size][kernel_size];

        for (unsigned int ch = 0; ch < input_shape.d; ch++)
        {
            if ((threadIdx.x < kernel_size) && (threadIdx.y < kernel_size))
            {
                unsigned int w_ofs = kernel_size*kernel_size*ch + filter*kernel_size*kernel_size*input_shape.d;
                w_shared[threadIdx.y][threadIdx.x] = w[w_ofs + threadIdx.y*kernel_size + threadIdx.x];
            }

            __syncthreads();

            if ((y <input_size_y) && (x < input_size_x))
            {
                unsigned int input_idx  = (ch*input_shape.h + y)*input_shape.w + x;

                if (kernel_size == 1)
                {
                    sum+= w_shared[0][0]*input[input_idx];
                }

                if (kernel_size == 3)
                {
                    sum+= w_shared[0][0]*input[input_idx]; input_idx++;
                    sum+= w_shared[0][1]*input[input_idx]; input_idx++;
                    sum+= w_shared[0][2]*input[input_idx]; input_idx++;
                    input_idx+= input_shape.w - kernel_size;

                    sum+= w_shared[1][0]*input[input_idx]; input_idx++;
                    sum+= w_shared[1][1]*input[input_idx]; input_idx++;
                    sum+= w_shared[1][2]*input[input_idx]; input_idx++;
                    input_idx+= input_shape.w - kernel_size;

                    sum+= w_shared[2][0]*input[input_idx]; input_idx++;
                    sum+= w_shared[2][1]*input[input_idx]; input_idx++;
                    sum+= w_shared[2][2]*input[input_idx]; input_idx++;
                    input_idx+= input_shape.w - kernel_size;
                }
            }

            __syncthreads();
        }

        unsigned int output_idx = (filter*input_shape.h + y + k_half)*input_shape.w + x + k_half;
        output[output_idx] = sum;
    }
}



template<unsigned int kernel_size>
__global__
void cuda_convolution_forward_kernel_1d(    float *output,
                                            float *input,
                                            float *w,
                                            float *bias,

                                            sShape output_shape,
                                            sShape input_shape,
                                            sShape kernel_shape
                                        )
{
    unsigned int x      = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int filter = threadIdx.y + blockIdx.y*blockDim.y;

    unsigned int k_half = (kernel_size - 1)/2;
    unsigned int input_size_x = input_shape.w - 2*k_half;

    __shared__ float w_shared[MAX_FEATURE_MAP_COUNT][kernel_size];
    if (threadIdx.x < kernel_size)
        for (unsigned int ch = 0; ch < input_shape.d; ch++)
        {
            unsigned int w_ofs = kernel_size*ch + filter*kernel_size*input_shape.d;
            w_shared[ch][threadIdx.x] = w[w_ofs + threadIdx.x];
        }

    __syncthreads();


    if (filter < output_shape.d)
    if (x < input_size_x)
    {
        float sum = bias[filter];

        unsigned int input_idx  = x;

        for (unsigned int ch = 0; ch < input_shape.d; ch++)
        {
            if (kernel_size == 1)
            {
                sum+= w_shared[ch][0]*input[input_idx]; input_idx++;
            }

            if (kernel_size == 3)
            {
                sum+= w_shared[ch][0]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][1]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][2]*input[input_idx]; input_idx++;
            }

            input_idx+= input_shape.w - kernel_size;
        }

        unsigned int output_idx = x + k_half + filter*input_shape.w;
        output[output_idx] = sum;
    }
}


__global__
void cuda_convolution_forward_kernel_any_size(  float *output,
                                                float *input,
                                                float *w,
                                                float *bias,

                                                sShape output_shape,
                                                sShape input_shape,
                                                sShape kernel_shape )
{
    unsigned int x      = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y      = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int z      = threadIdx.z + blockIdx.z*blockDim.z;

    unsigned int filter = z;

    unsigned int k_half_h = (kernel_shape.h - 1)/2;
    unsigned int k_half_w = (kernel_shape.w - 1)/2;

    unsigned int input_size_y = input_shape.h - 2*k_half_h;
    unsigned int input_size_x = input_shape.w - 2*k_half_w;

    if (filter < output_shape.d)
    if (y < input_size_y)
    if (x < input_size_x)
    {
        float sum = bias[filter];

        for (unsigned int ch = 0; ch < input_shape.d; ch++)
        {
            unsigned int filter_idx = (filter*input_shape.d + ch)*kernel_shape.h*kernel_shape.w;
            unsigned int input_idx  = (ch*input_shape.h + y)*input_shape.w + x;

            for (unsigned int ky = 0; ky < kernel_shape.h; ky++)
            {
                for (unsigned int kx = 0; kx < kernel_shape.w; kx++)
                {
                    unsigned int input_idx = (ch*input_shape.h + y + ky)*input_shape.w + x + kx;
                    sum+= w[filter_idx]*input[input_idx];
                    filter_idx++;
                    input_idx++;
                }

                input_idx+= input_shape.w - kernel_shape.w;
            }

        }

        unsigned int output_idx = (filter*input_shape.h + y + k_half_h)*input_shape.w + x + k_half_w;
        output[output_idx] = sum;
    }
}


void convolution_layer_forward(   Tensor &output, Tensor &input,
                                  Tensor &w, Tensor &bias)
{
      unsigned int input_size_y = input.h() - w.h();
      unsigned int input_size_x = input.w() - w.w();

      sShape input_shape;

      input_shape.w = input.w();
      input_shape.h = input.h();
      input_shape.d = input.d();

      sShape output_shape;

      output_shape.w = output.w();
      output_shape.h = output.h();
      output_shape.d = output.d();

      sShape kernel_shape;

      kernel_shape.w = w.w();
      kernel_shape.h = w.h();
      kernel_shape.d = output_shape.d;


    #ifdef NETWORK_USE_CUDA

        if ((kernel_shape.w == 1)&&(kernel_shape.h == 1))
        {
            dim3 block(16, 16, 1);
            dim3 grid( (input_size_x      + block.x + 1)/block.x,
                       (input_size_y      + block.y + 1)/block.y,
                       (kernel_shape.d + block.z + 1)/block.z );

            cuda_convolution_forward_kernel<1><<<grid, block>>>( output.v,
                                                                          input.v,
                                                                          w.v,
                                                                          bias.v,

                                                                          output_shape,
                                                                          input_shape,
                                                                          kernel_shape);
        }
        else if ((kernel_shape.w == 3)&&(kernel_shape.h == 3))
        {
            dim3 block(16, 16, 1);
            dim3 grid( (input_size_x      + block.x + 1)/block.x,
                       (input_size_y      + block.y + 1)/block.y,
                       (kernel_shape.d + block.z + 1)/block.z );

            cuda_convolution_forward_kernel<3><<<grid, block>>>( output.v,
                                                                          input.v,
                                                                          w.v,
                                                                          bias.v,

                                                                          output_shape,
                                                                          input_shape,
                                                                          kernel_shape);
        }
        else if ((kernel_shape.w == 3)&&(kernel_shape.h == 1))
        {
            dim3 block(16, 1);
            dim3 grid( (input_size_x      + block.x + 1)/block.x,
                       (kernel_shape.d + block.y + 1)/block.y );

            cuda_convolution_forward_kernel_1d<3><<<grid, block>>>( output.v,
                                                                    input.v,
                                                                    w.v,
                                                                    bias.v,

                                                                    output_shape,
                                                                    input_shape,
                                                                    kernel_shape);
        }
        else
        {
            dim3 block(16, 16, 1);
            dim3 grid( (input_size_x      + block.x + 1)/block.x,
                       (input_size_y      + block.y + 1)/block.y,
                       (kernel_shape.d + block.z + 1)/block.z );

            cuda_convolution_forward_kernel_any_size<<<grid, block>>>(
                                                                        output.v,
                                                                        input.v,
                                                                        w.v,
                                                                        bias.v,

                                                                        output_shape,
                                                                        input_shape,
                                                                        kernel_shape);
          }

          cudaDeviceSynchronize();

    #else

      cpu_convolution_forward_kernel( output.v,
                                      input.v,
                                      w.v,
                                      bias.v,

                                      output_shape,
                                      input_shape,
                                      kernel_shape);
    #endif
}
