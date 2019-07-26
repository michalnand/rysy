#include "convolution_layer_backward.cuh"


__host__
void cpu_convolution_layer_weights_gradient(  float *w_grad,
                                              float *error,
                                              float *input,


                                              sShape error_shape,
                                              sShape input_shape,
                                              sShape kernel_shape )
{
  unsigned int input_size_y = input_shape.h - kernel_shape.h;
  unsigned int input_size_x = input_shape.w - kernel_shape.w;

  for (unsigned int kd = 0; kd < kernel_shape.d; kd++)
  for (unsigned int ky = 0; ky < kernel_shape.h; ky++)
  for (unsigned int kx = 0; kx < kernel_shape.w; kx++)
  {
    unsigned int filter = kd/input_shape.d;
    unsigned int ch     = kd%input_shape.d;
    unsigned int w_ofs  = (kd*kernel_shape.h + ky)*kernel_shape.w + kx;

    float w_dif = 0.0;
    for (unsigned int y = 0; y < input_size_y; y++)
    {
      unsigned int input_idx  = (ch*input_shape.h + y + ky)*input_shape.w + kx;
      unsigned int output_idx = (filter*error_shape.h + y)*error_shape.w;

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

/*
__global__
void cuda_convolution_layer_weights_gradient_any_size(      float *w_grad,
                                                            float *error,
                                                            float *input,

                                                            sShape error_shape,
                                                            sShape input_shape,
                                                            sShape kernel_shape )
{
    unsigned int kx  = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int ky  = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int kd  = threadIdx.z + blockIdx.z*blockDim.z;

    if (kd < kernel_shape.d)
    if (ky < kernel_shape.h)
    if (kx < kernel_shape.w)
    {
          unsigned int kw_half      = (kernel_shape.w - 1)/2;
          unsigned int kh_half      = (kernel_shape.h - 1)/2;
          unsigned int input_size_x = input_shape.w - kw_half;
          unsigned int input_size_y = input_shape.h - kh_half;

          unsigned int filter = kd/input_shape.d;
          unsigned int ch     = kd%input_shape.d;
          unsigned int w_ofs  = (kd*kernel_shape.h + ky)*kernel_shape.w + kx;

          float w_dif = 0.0;

          for (unsigned int y = 0; y < input_size_y; y++)
          {
             unsigned int input_idx  = (ch*input_shape.h + y + ky)*input_shape.w + kx;
             unsigned int error_idx  = (filter*error_shape.h + y + kh_half)*error_shape.w + kw_half;

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
*/


template<unsigned int kernel_size>
__global__
void cuda_convolution_layer_weights_gradient(      float *w_grad,
                                                   const float *error,
                                                   const float *input,

                                                   const sShape error_shape,
                                                   const sShape input_shape)
{
    unsigned int kernel     = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int channel    = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int k_idx      = threadIdx.z + blockIdx.z*blockDim.z;

    if (kernel < error_shape.d)
    if (channel < input_shape.d)
    if (k_idx < kernel_size*kernel_size)
    {
        unsigned int kx = k_idx%kernel_size;
        unsigned int ky = k_idx/kernel_size;

        const unsigned int k_half      = (kernel_size - 1)/2;

        float sum = 0.0;
 
        for (unsigned int y = k_half; y < input_shape.h - k_half; y++)
        {
            unsigned int input_idx  = (channel*input_shape.h + y + ky)*input_shape.w + kx;
            unsigned int error_idx  = (kernel*error_shape.h + y + k_half)*error_shape.w + k_half;

            for (unsigned int x = k_half; x < input_shape.w - k_half; x++)
            {
                sum+= input[input_idx]*error[error_idx];
                input_idx++;
                error_idx++;
            }
        }

        unsigned int w_idx  = ((kernel*input_shape.d + channel)*kernel_size + ky)*kernel_size + kx;
        w_grad[w_idx]+= sum;
      }
}


__global__
void cuda_convolution_layer_weights_gradient_any_size(      float *w_grad,
                                                            float *error,
                                                            float *input,

                                                            sShape error_shape,
                                                            sShape input_shape,
                                                            sShape kernel_shape )
{
    unsigned int kernel     = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int channel    = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int k_idx      = threadIdx.z + blockIdx.z*blockDim.z;

    if (kernel < error_shape.d)
    if (channel < input_shape.d)
    if (k_idx < kernel_shape.w*kernel_shape.h)
    {
        unsigned int kx = k_idx%kernel_shape.w;
        unsigned int ky = k_idx/kernel_shape.w;

        unsigned int kw_half      = (kernel_shape.w - 1)/2;
        unsigned int kh_half      = (kernel_shape.h - 1)/2;

        float sum = 0.0;

        for (unsigned int y = kh_half; y < input_shape.h - kh_half; y++)
        {
            unsigned int input_idx  = (channel*input_shape.h + y + ky)*input_shape.w + kx;
            unsigned int error_idx  = (kernel*error_shape.h + y + kh_half)*error_shape.w + kw_half;

            for (unsigned int x = kw_half; x < input_shape.w - kw_half; x++)
            {
                sum+= input[input_idx]*error[error_idx];
                input_idx++;
                error_idx++;
            }
        }

        unsigned int w_idx  = ((kernel*input_shape.d + channel)*kernel_shape.h + ky)*kernel_shape.w + kx;
        w_grad[w_idx]+= sum;
      }
}


void convolution_layer_gradient(Tensor &w_grad, Tensor &input, Tensor &error)
{
    sShape input_shape;

    input_shape.w = input.w();
    input_shape.h = input.h();
    input_shape.d = input.d();

    sShape error_shape;

    error_shape.w = error.w();
    error_shape.h = error.h();
    error_shape.d = error.d();

    sShape weights_shape;

    weights_shape.w = w_grad.w();
    weights_shape.h = w_grad.h();
    weights_shape.d = w_grad.d();

    #ifdef NETWORK_USE_CUDA

        if ((weights_shape.w == 3)&&(weights_shape.h == 3))
        {
            dim3 block(8, 8, 1);
            dim3 grid( (error_shape.d  + block.x + 1)/block.x,
                       (input_shape.d  + block.y + 1)/block.y,
                       10);

            cuda_convolution_layer_weights_gradient<3><<<grid, block>>>(   w_grad.v,
                                                                           error.v,
                                                                           input.v,

                                                                           error_shape,
                                                                           input_shape);

            cudaDeviceSynchronize();
        }
        else
        {
            dim3 block(8, 8, 4);
            dim3 grid( (error_shape.d  + block.x + 1)/block.x,
                       (input_shape.d  + block.y + 1)/block.y,
                       (weights_shape.w*weights_shape.h  + block.z + 1)/block.z);

            cuda_convolution_layer_weights_gradient_any_size<<<grid, block>>>(   w_grad.v,
                                                                        error.v,
                                                                        input.v,

                                                                        error_shape,
                                                                        input_shape,
                                                                        weights_shape);

            cudaDeviceSynchronize();
        }

            /*
            {
                dim3 block(4, 4, 8);
                dim3 grid( (weights_shape.w  + block.x + 1)/block.x,
                           (weights_shape.h  + block.y + 1)/block.y,
                           (weights_shape.d  + block.z + 1)/block.z);

                cuda_convolution_layer_weights_gradient_any_size<<<grid, block>>>(   w_grad.v,
                                                                            error.v,
                                                                            input.v,

                                                                            error_shape,
                                                                            input_shape,
                                                                            weights_shape);

                cudaDeviceSynchronize();
            }
            */
  #else

    cpu_convolution_layer_weights_gradient( w_grad.v,
                                            error.v,
                                            input.v,

                                            error_shape,
                                            input_shape,
                                            weights_shape );

  #endif
}









__host__
void cpu_convolution_learn_bias_kernel(  float *bias,
                                         float *error,
                                         sShape output_shape,
                                         float learning_rate,
                                         unsigned int size)
{
  for (unsigned int filter = 0; filter < output_shape.d; filter++)
  {
    float sum = 0.0;

    for (unsigned int y = 0; y < output_shape.h; y++)
      for (unsigned int x = 0; x < output_shape.w; x++)
      {
        unsigned int output_idx = (filter*output_shape.h + y)*output_shape.w + x;
        sum+= error[output_idx];
      }

    bias[filter]+= learning_rate*sum;
  }
}

__global__
void cuda_convolution_learn_bias_kernel(  float *bias,
                                          float *error,
                                          sShape output_shape,
                                          float learning_rate,
                                          unsigned int size)
{
  unsigned int x  = threadIdx.x + blockIdx.x*blockDim.x;
  unsigned int y  = threadIdx.y + blockIdx.y*blockDim.y;

  if (y < output_shape.h)
  if (x < output_shape.w)
  {
    for (unsigned int filter = 0; filter < size; filter++)
    {
      unsigned int error_idx = (filter*output_shape.h + y)*output_shape.w + x;

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
                                                              error.shape_struct(),
                                                              learning_rate,
                                                              bias.size());
        cudaDeviceSynchronize();
      }

  #else

      cpu_convolution_learn_bias_kernel(  bias.v,
                                          error.v,
                                          error.shape_struct(),
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


//OLD implementations - SLOW
/*
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

    if (    (ch < channels_count) &&
            (y <  height_) &&
            (x <  width_) )
    {
        for (unsigned int kernel = 0; kernel < kernels_count; kernel++)
        {
          unsigned int w_idx     = kernel*channels_count*kernel_size*kernel_size + ch*kernel_size*kernel_size;
          unsigned int error_idx = (kernel*height + y + kernel_size_half)*width + x + kernel_size_half;

          float err = error[error_idx];

          unsigned int error_back_idx = (ch*height + y)*width + x;

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
      }
    }
}
*/






__global__
void cuda_convolution_back_kernel_1(    float *error,
                                        float *error_back,
                                        float *w,

                                        unsigned int width,
                                        unsigned int height,
                                        unsigned int channels_count,
                                        unsigned int kernels_count )
{
    unsigned int x       = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y       = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int channel = threadIdx.z + blockIdx.z*blockDim.z;


    if (    (channel < channels_count) &&
            (y <  height) &&
            (x <  width) )
    {
        unsigned int w_idx      = channel;
        unsigned int error_idx  = y*width + x;

        float sum = 0.0;

        for (unsigned int kernel = 0; kernel < kernels_count; kernel++)
        {
            sum+=       w[w_idx]*error[error_idx];

            w_idx+=     channels_count;
            error_idx+= width*height;
        }

        unsigned int error_back_idx = (channel*height + y)*width + x;
        error_back[error_back_idx]  = sum;
    }
}



__global__
void cuda_convolution_back_kernel_3(    float *error,
                                        float *error_back,
                                        float *w,

                                        int width,
                                        int height,
                                        unsigned int channels_count,
                                        unsigned int kernels_count )
{
    unsigned int x       = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y       = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int channel = threadIdx.z + blockIdx.z*blockDim.z;

    unsigned int width_         = width - 1;
    unsigned int height_        = height - 1;
    unsigned int kernel_size    = 3;

    //without shared memory implementation


    if ( (channel < channels_count) && (y < height) && (x < width))
    {
        float sum = 0.0;

        for (unsigned int kernel = 0; kernel < kernels_count; kernel++)
        {
            unsigned int w_idx        = (kernel*channels_count + channel)*kernel_size*kernel_size;
            unsigned int error_offset = (kernel*height + y + 0)*width + x + 0;

            if ((x < width_) && (y < height_))
                sum+= w[w_idx]*error[error_offset + 1*width + 1]; w_idx++;

            if (y < height_)
                sum+= w[w_idx]*error[error_offset + 1*width + 0]; w_idx++;

            if ((x > 0) && (y < height_))
                sum+= w[w_idx]*error[error_offset + 1*width - 1]; w_idx++;



            if (x < width_)
                sum+= w[w_idx]*error[error_offset + 0*width + 1]; w_idx++;

            if (true)
                sum+= w[w_idx]*error[error_offset + 0*width + 0]; w_idx++;

            if (x > 0)
                sum+= w[w_idx]*error[error_offset + 0*width - 1]; w_idx++;



            if ((x < width_) && (y > 0))
                sum+= w[w_idx]*error[error_offset - 1*width + 1]; w_idx++;

            if (y > 0)
                sum+= w[w_idx]*error[error_offset - 1*width + 0]; w_idx++;

            if ((x > 0) && (y > 0))
                sum+= w[w_idx]*error[error_offset - 1*width - 1]; w_idx++;
        }

        unsigned int error_back_idx = (channel*height + y)*width + x;
        error_back[error_back_idx]  = sum;
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



void convolution_layer_backward( Tensor &error_back, Tensor &error, Tensor &w)
{
    unsigned int kernel_width = w.w();
    unsigned int kernel_height = w.h();

    #ifdef NETWORK_USE_CUDA

    if ((kernel_width == 1) && (kernel_height == 1))
    {
        dim3 block(8, 8, 4);
        dim3 grid( (error_back.w()      + block.x + 1)/block.x,
                   (error_back.h()      + block.y + 1)/block.y,
                   (error_back.d()      + block.z + 1)/block.z );

        cuda_convolution_back_kernel_1<<<grid, block>>>(    error.v,
                                                            error_back.v,
                                                            w.v,

                                                            error.w(),
                                                            error.h(),
                                                            error_back.d(),
                                                            error.d() );
    }
    else if ((kernel_width == 3) && (kernel_height == 3))
    {
        dim3 block(8, 8, 8);
        dim3 grid( (error_back.w()      + block.x + 1)/block.x,
                   (error_back.h()      + block.y + 1)/block.y,
                   (error_back.d()      + block.z + 1)/block.z );

        cuda_convolution_back_kernel_3<<<grid, block>>>(    error.v,
                                                            error_back.v,
                                                            w.v,

                                                            error.w(),
                                                            error.h(),
                                                            error_back.d(),
                                                            error.d() );

/*
        error_back.clear();

      dim3 block(4, 4, 16);
      dim3 grid( (error_back.w()      + block.x + 1)/block.x,
                 (error_back.h()      + block.y + 1)/block.y,
                 (error_back.d()      + block.z + 1)/block.z );

      cuda_convolution_back_kernel<3><<<grid, block>>>( error.v,
                                                        error_back.v,
                                                        w.v,

                                                        error.w(),
                                                        error.h(),
                                                        error_back.d(),
                                                        error.d() );
*/
    }
    else
    {
        error_back.clear();

        dim3 block(4, 4, 16);
        dim3 grid( (error_back.w()      + block.x + 1)/block.x,
                   (error_back.h()      + block.y + 1)/block.y,
                   (error_back.d()      + block.z + 1)/block.z );

        cuda_convolution_back_kernel_any_size<<<grid, block>>>( error.v,
                                                                error_back.v,
                                                                w.v,

                                                                error.w(),
                                                                error.h(),
                                                                error_back.d(),
                                                                error.d(),
                                                                w.w(),
                                                                w.h());
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
