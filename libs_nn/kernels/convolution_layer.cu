#include "convolution_layer.cuh"

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
    for (unsigned int y = 0; y <= input_size_y; y++)
      for (unsigned int x = 0; x <= input_size_x; x++)
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
  unsigned int filter = threadIdx.z + blockIdx.z*blockDim.z;


  unsigned int k_half = (kernel_size - 1)/2;

  unsigned int input_size_y = input_geometry.h - 2*k_half;
  unsigned int input_size_x = input_geometry.w - 2*k_half;


  if (filter < kernel_geometry.d)
  if (y <= input_size_y)
  if (x <= input_size_x)
  {
    unsigned int filter_idx = kernel_size*kernel_size*input_geometry.d*filter;

    float sum = 0.0;

    for (unsigned int ch = 0; ch < input_geometry.d; ch++)
    {
      unsigned int input_idx = (ch*input_geometry.h + y)*input_geometry.w + x;

      if (kernel_size == 1)
      {
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;
      }

      if (kernel_size == 3)
      {
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;

        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;

        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;
      }

      if (kernel_size == 5)
      {
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;

        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;

        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;

        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;

        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;
      }

      if (kernel_size == 7)
      {
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;

        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;

        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;

        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;

        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;

        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;

        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        sum+= w[filter_idx]*input[input_idx]; filter_idx++; input_idx++;
        input_idx+= input_geometry.w - kernel_size;
      }

    }


    unsigned int output_idx = (filter*input_geometry.h + y + k_half)*input_geometry.w + x + k_half;
    sum+= bias[filter];
    output[output_idx] = sum;
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
  unsigned int filter = threadIdx.z + blockIdx.z*blockDim.z;


  unsigned int k_half_w = (kernel_geometry.w - 1)/2;
  unsigned int k_half_h = (kernel_geometry.h - 1)/2;

  unsigned int input_size_y = input_geometry.h - 2*k_half_h;
  unsigned int input_size_x = input_geometry.w - 2*k_half_w;


  if (filter < kernel_geometry.d)
  if (y <= input_size_y)
  if (x <= input_size_x)
  {
    unsigned int filter_idx = kernel_geometry.h*kernel_geometry.w*input_geometry.d*filter;

    float sum = 0.0;

    for (unsigned int ch = 0; ch < input_geometry.d; ch++)
      for (unsigned int ky = 0; ky < kernel_geometry.h; ky++)
        for (unsigned int kx = 0; kx < kernel_geometry.w; kx++)
        {
          unsigned int input_idx = (ch*input_geometry.h + y + ky)*input_geometry.w + x + kx;
          sum+= w[filter_idx]*input[input_idx];
          filter_idx++;
        }


    unsigned int output_idx = (filter*input_geometry.h + y + k_half_h)*input_geometry.w + x + k_half_w;
    sum+= bias[filter];
    output[output_idx] = sum;
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

      dim3 block(8, 8, 8);
      dim3 grid( (input_size_x      + block.x - 1)/block.x,
                 (input_size_y      + block.y - 1)/block.y,
                 (kernel_geometry.d + block.z - 1)/block.z );

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

                case 7:  cuda_convolution_forward_kernel<7><<<grid, block>>>( output.v,
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






__host__
void cpu_convolution_layer_weights_gradient(  float *w_grad,
                                              float *error,
                                              float *input,


                                              sGeometry error_geometry,
                                              sGeometry input_geometry,
                                              sGeometry kernel_geometry )
{
  unsigned int input_size_y = input_geometry.h - kernel_geometry.h;
  unsigned int input_size_x = input_geometry.w - kernel_geometry.w;

  for (unsigned int kd = 0; kd < kernel_geometry.d; kd++)
  for (unsigned int ky = 0; ky < kernel_geometry.h; ky++)
  for (unsigned int kx = 0; kx < kernel_geometry.w; kx++)
  {
    unsigned int filter = kd/input_geometry.d;
    unsigned int ch     = kd%input_geometry.d;
    unsigned int w_ofs  = (kd*kernel_geometry.h + ky)*kernel_geometry.w + kx;

    float w_dif = 0.0;
    for (unsigned int y = 0; y < input_size_y; y++)
    {
      unsigned int input_idx  = (ch*input_geometry.h + y + ky)*input_geometry.w + kx;
      unsigned int output_idx = (filter*error_geometry.h + y)*error_geometry.w;

      for (unsigned int x = 0; x < input_size_x; x++)
      {
        w_dif+= error[output_idx]*input[input_idx];
        output_idx++;
        input_idx++;
      }
    }

    w_grad[w_ofs]+= w_dif;
  }
}


__global__
void cuda_convolution_layer_weights_gradient(   float *w_grad,
                                                float *error,
                                                float *input,

                                                sGeometry error_geometry,
                                                sGeometry input_geometry,
                                                sGeometry kernel_geometry )
{
  unsigned int kx  = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int ky  = threadIdx.y + blockIdx.y*blockDim.y;
  unsigned int kd  = threadIdx.z + blockIdx.z*blockDim.z;

  if (kd < kernel_geometry.d)
  if (ky < kernel_geometry.h)
  if (kx < kernel_geometry.w)
  {
      unsigned int kw_half      = (kernel_geometry.w - 1)/2;
      unsigned int kh_half      = (kernel_geometry.h - 1)/2;
      unsigned int input_size_y = input_geometry.h - kh_half;
      unsigned int input_size_x = input_geometry.w - kw_half;

      unsigned int filter = kd/input_geometry.d;
      unsigned int ch     = kd%input_geometry.d;
      unsigned int w_ofs  = (kd*kernel_geometry.h + ky)*kernel_geometry.w + kx;

      float w_dif = 0.0;

      for (unsigned int y = 0; y < input_size_y; y++)
      {
        unsigned int input_idx  = (ch*input_geometry.h + y + ky)*input_geometry.w + kx;
        unsigned int error_idx = (filter*error_geometry.h + y + kh_half)*error_geometry.w + kw_half;

        for (unsigned int x = 0; x < input_size_x; x++)
        {
          w_dif+= error[error_idx]*input[input_idx];
          error_idx++;
          input_idx++;
        }
      }

      w_grad[w_ofs]+= w_dif;
  }
}


void convolution_layer_gradient(Tensor &w_grad, Tensor &input, Tensor &error)
{
  sGeometry input_geometry;

  input_geometry.w = input.w();
  input_geometry.h = input.h();
  input_geometry.d = input.d();

  sGeometry error_geometry;

  error_geometry.w = error.w();
  error_geometry.h = error.h();
  error_geometry.d = error.d();

  sGeometry weights_geometry;

  weights_geometry.w = w_grad.w();
  weights_geometry.h = w_grad.h();
  weights_geometry.d = w_grad.d();


  #ifdef NETWORK_USE_CUDA

        dim3 block(8, 8, 8);
        dim3 grid( (weights_geometry.w  + block.x - 1)/block.x,
                   (weights_geometry.h  + block.y - 1)/block.y,
                   (weights_geometry.d  + block.z - 1)/block.z);

        cuda_convolution_layer_weights_gradient<<<grid, block>>>(   w_grad.v,
                                                                    error.v,
                                                                    input.v,

                                                                    error_geometry,
                                                                    input_geometry,
                                                                    weights_geometry );

        cudaDeviceSynchronize();


  #else

    cpu_convolution_layer_weights_gradient( w_grad.v,
                                            error.v,
                                            input.v,

                                            error_geometry,
                                            input_geometry,
                                            weights_geometry );

  #endif
}









__host__
void cpu_convolution_learn_bias_kernel(  float *bias,
                                         float *error,
                                         sGeometry output_geometry,
                                         float learning_rate,
                                         unsigned int size)
{
  for (unsigned int filter = 0; filter < output_geometry.d; filter++)
  {
    float sum = 0.0;

    for (unsigned int y = 0; y < output_geometry.h; y++)
      for (unsigned int x = 0; x < output_geometry.w; x++)
      {
        unsigned int output_idx = (filter*output_geometry.h + y)*output_geometry.w + x;
        sum+= error[output_idx];
      }

    bias[filter]+= learning_rate*sum;
  }
}

__global__
void cuda_convolution_learn_bias_kernel(  float *bias,
                                          float *error,
                                          sGeometry output_geometry,
                                          float learning_rate,
                                          unsigned int size)
{
  unsigned int x  = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int y  = threadIdx.y + blockIdx.y*blockDim.y;

  if (y < output_geometry.h)
  if (x < output_geometry.w)
  {
    for (unsigned int filter = 0; filter < size; filter++)
    {
      unsigned int error_idx = (filter*output_geometry.h + y)*output_geometry.w + x;

      float b_dif = learning_rate*error[error_idx];

      atomicAdd(&bias[filter], b_dif);
    }
  }
}

void convolution_layer_update_bias(Tensor &bias, Tensor &error, float learning_rate)
{
  #ifdef NETWORK_USE_CUDA

      {
        dim3 block(8, 8);
        dim3 grid( (error.w()  + block.x - 1)/block.x,
                   (error.h()  + block.y - 1)/block.y);

        cuda_convolution_learn_bias_kernel<<<grid, block>>>(  bias.v,
                                                              error.v,
                                                              error.get_geometry(),
                                                              learning_rate,
                                                              bias.size());
        cudaDeviceSynchronize();
      }

  #else

      cpu_convolution_learn_bias_kernel(  bias.v,
                                          error.v,
                                          error.get_geometry(),
                                          learning_rate,
                                          bias.size());

  #endif
}



__host__
void cpu_convolution_back_kernel( float *error,
                                  float *error_back,
                                  float *w,

                                  unsigned int width,
                                  unsigned int height,
                                  unsigned int channels_count,
                                  unsigned int kernels_count,
                                  unsigned int kw,
                                  unsigned int kh)
{
  unsigned int kw_half = (kw - 1)/2;
  unsigned int kh_half = (kh - 1)/2;

  unsigned int height_  = height - 2*kh_half;
  unsigned int width_   = width  - 2*kw_half;


  for (unsigned int ch = 0; ch < channels_count; ch++)
  for (unsigned int y = 0; y < height_; y++)
  for (unsigned int x = 0; x < width_; x++)
  {
    for (unsigned int kernel = 0; kernel < kernels_count; kernel++)
    {
      unsigned int w_idx     = kernel*channels_count*kh*kw + ch*kh*kw;
      unsigned int error_idx = (kernel*height + y + kh_half)*width + x + kw_half;

      float err = error[error_idx];

      unsigned int error_back_idx = (ch*height + y)*width + x;


      for (unsigned int ky = 0; ky < kh; ky++)
      {
        for (unsigned int kx = 0; kx < kw; kx++)
        {
          error_back[error_back_idx]+= err*w[w_idx];

          w_idx++;
          error_back_idx++;
        }

        error_back_idx+= width - kw;
      }
    }
  }
}

template<const unsigned int kernel_size>
__global__
void cuda_convolution_back_kernel(  float *error,
                                    float *error_back,
                                    float *w,

                                    unsigned int width,
                                    unsigned int height,
                                    unsigned int channels_count,
                                    unsigned int kernels_count )
{
  unsigned int x      = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int y      = threadIdx.y + blockIdx.y*blockDim.y;
  unsigned int ch     = threadIdx.z + blockIdx.z*blockDim.z;

  unsigned int kernel_size_half = (kernel_size - 1)/2;

  unsigned int height_  = height - 2*kernel_size_half;
  unsigned int width_   = width  - 2*kernel_size_half;

  if (  (ch < channels_count) &&
        (y <  height_) &&
        (x <  width_) )
  {
    for (unsigned int kernel = 0; kernel < kernels_count; kernel++)
    {
      unsigned int w_idx     = kernel*channels_count*kernel_size*kernel_size + ch*kernel_size*kernel_size;
      unsigned int error_idx = (kernel*height + y + kernel_size_half)*width + x + kernel_size_half;

      float err = error[error_idx];

      unsigned int error_back_idx = (ch*height + y)*width + x;

      /*
      for (unsigned int ky = 0; ky < kernel_size; ky++)
      {
        for (unsigned int kx = 0; kx < kernel_size; kx++)
        {
          atomicAdd(&error_back[error_back_idx], err*w[w_idx]);

          w_idx++;
          error_back_idx++;
        }

        error_back_idx+= width - kernel_size;
      }
      */

      if (kernel_size == 1)
      {
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;
      }

      if (kernel_size == 3)
      {
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;

        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;

        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;
      }

      if (kernel_size == 5)
      {
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;

        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;

        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;

        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;

        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;
      }



      if (kernel_size == 7)
      {
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;

        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;

        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;

        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;

        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;

        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;

        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        atomicAdd(&error_back[error_back_idx], err*w[w_idx]); w_idx++; error_back_idx++;
        error_back_idx+= width - kernel_size;
      }

    }
  }
}







__global__
void cuda_convolution_back_kernel_any_size(   float *error,
                                              float *error_back,
                                              float *w,

                                              unsigned int width,
                                              unsigned int height,
                                              unsigned int channels_count,
                                              unsigned int kernels_count,
                                              unsigned int kw,
                                              unsigned int kh)
{
  unsigned int x      = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int y      = threadIdx.y + blockIdx.y*blockDim.y;
  unsigned int ch     = threadIdx.z + blockIdx.z*blockDim.z;

  unsigned int kw_half = (kw - 1)/2;
  unsigned int kh_half = (kh - 1)/2;

  unsigned int height_  = height - 2*kh_half;
  unsigned int width_   = width  - 2*kw_half;

  if (  (ch < channels_count) &&
        (y <  height_) &&
        (x <  width_) )
  {
    for (unsigned int kernel = 0; kernel < kernels_count; kernel++)
    {
      unsigned int w_idx     = kernel*channels_count*kw*kh + ch*kw*kh;
      unsigned int error_idx = (kernel*height + y + kh_half)*width + x + kw_half;

      float err = error[error_idx];

      unsigned int error_back_idx = (ch*height + y)*width + x;

      for (unsigned int ky = 0; ky < kh; ky++)
      {
        for (unsigned int kx = 0; kx < kw; kx++)
        {
          atomicAdd(&error_back[error_back_idx], err*w[w_idx]);

          w_idx++;
          error_back_idx++;
        }

        error_back_idx+= width - kw;
      }
    }
  }
}

void convolution_layer_backward( Tensor &error_back, Tensor &input, Tensor &error, Tensor &w)
{
  error_back.clear();

  unsigned int kernel_size = w.w();

  #ifdef NETWORK_USE_CUDA

      dim3 block(4, 4, 16);
      dim3 grid( (error_back.w()      + block.x - 1)/block.x,
                 (error_back.h()      + block.y - 1)/block.y,
                 (error_back.d()      + block.z - 1)/block.z );

      switch (kernel_size)
      {
        case 1:       cuda_convolution_back_kernel<1><<<grid, block>>>( error.v,
                                                                        error_back.v,
                                                                        w.v,

                                                                        error.w(),
                                                                        error.h(),
                                                                        error_back.d(),
                                                                        error.d() );
                      break;

        case 3:      cuda_convolution_back_kernel<3><<<grid, block>>>( error.v,
                                                                       error_back.v,
                                                                       w.v,

                                                                       error.w(),
                                                                       error.h(),
                                                                       error_back.d(),
                                                                       error.d() );
                      break;

        case 5:       cuda_convolution_back_kernel<5><<<grid, block>>>( error.v,
                                                                        error_back.v,
                                                                        w.v,

                                                                        error.w(),
                                                                        error.h(),
                                                                        error_back.d(),
                                                                        error.d() );
                      break;

        case 7:       cuda_convolution_back_kernel<7><<<grid, block>>>( error.v,
                                                                        error_back.v,
                                                                        w.v,

                                                                        error.w(),
                                                                        error.h(),
                                                                        error_back.d(),
                                                                        error.d() );

                      break;

        default:
                    cuda_convolution_back_kernel_any_size<<<grid, block>>>( error.v,
                                                                            error_back.v,
                                                                            w.v,

                                                                            error.w(),
                                                                            error.h(),
                                                                            error_back.d(),
                                                                            error.d(),
                                                                            w.w(),
                                                                            w.h());

                    break;
      }

      cudaDeviceSynchronize();


  #else

    cpu_convolution_back_kernel( error.v,
                                 error_back.v,
                                 w.v,

                                 error.w(),
                                 error.h(),
                                 error_back.d(),
                                 error.d(),
                                 w.w(),
                                 w.h());

  #endif
}
