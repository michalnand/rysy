#include "convolution_layer_forward.cuh"


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

    unsigned int filter = z;

    unsigned int k_half = (kernel_size - 1)/2;
    unsigned int input_size_y = input_geometry.h - 2*k_half;
    unsigned int input_size_x = input_geometry.w - 2*k_half;

    __shared__ float w_shared[320][kernel_size][kernel_size];
    if ( (threadIdx.x < kernel_size) && (threadIdx.y < kernel_size) )
    for (unsigned int ch = 0; ch < input_geometry.d; ch++)
    {
        unsigned int w_ofs = kernel_size*kernel_size*ch + filter*kernel_size*kernel_size*input_geometry.d;
        w_shared[ch][threadIdx.y][threadIdx.x] = w[w_ofs + threadIdx.y*kernel_size + threadIdx.x];
    }

    __syncthreads();


    if (filter < output_geometry.d)
    if (y < input_size_y)
    if (x < input_size_x)
    {
        float sum = bias[filter];

        for (unsigned int ch = 0; ch < input_geometry.d; ch++)
        {
            unsigned int input_idx  = (ch*input_geometry.h + y)*input_geometry.w + x;

            if (kernel_size == 1)
            {
                sum+= w_shared[ch][0][0]*input[input_idx];
                input_idx+= input_geometry.w;
            }

            if (kernel_size == 3)
            {
                sum+= w_shared[ch][0][0]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][0][1]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][0][2]*input[input_idx]; input_idx++;
                input_idx+= input_geometry.w - kernel_size;

                sum+= w_shared[ch][1][0]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][1][1]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][1][2]*input[input_idx]; input_idx++;
                input_idx+= input_geometry.w - kernel_size;

                sum+= w_shared[ch][2][0]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][2][1]*input[input_idx]; input_idx++;
                sum+= w_shared[ch][2][2]*input[input_idx]; input_idx++;
                input_idx+= input_geometry.w - kernel_size;
            }

        }

        //apply ReLU
        if (sum < 0.0)
            sum = 0.0;

        unsigned int output_idx = (filter*input_geometry.h + y + k_half)*input_geometry.w + x + k_half;
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
    unsigned int z      = threadIdx.z + blockIdx.z*blockDim.z;

    unsigned int filter = z;

    unsigned int k_half_h = (kernel_geometry.h - 1)/2;
    unsigned int k_half_w = (kernel_geometry.w - 1)/2;

    unsigned int input_size_y = input_geometry.h - 2*k_half_h;
    unsigned int input_size_x = input_geometry.w - 2*k_half_w;

    if (filter < output_geometry.d)
    if (y < input_size_y)
    if (x < input_size_x)
    {
        float sum = bias[filter];

        for (unsigned int ch = 0; ch < input_geometry.d; ch++)
        {
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

        }

        unsigned int output_idx = (filter*input_geometry.h + y + k_half_h)*input_geometry.w + x + k_half_w;
        output[output_idx] = sum; 
    }
}


void convolution_layer_forward( float *output, float *input,
                                float *weights, float *bias,
                                sGeometry input_geometry,
                                sGeometry kernel_geometry,
                                sGeometry output_geometry )
{
    unsigned int input_size_y = input_geometry.h - kernel_geometry.h;
    unsigned int input_size_x = input_geometry.w - kernel_geometry.w;

    kernel_geometry.d = output_geometry.d;

    dim3 block(16, 16, 1);
    dim3 grid(  (input_size_x      + block.x + 1)/block.x,
                (input_size_y      + block.y + 1)/block.y,
                (kernel_geometry.d + block.z + 1)/block.z );

    unsigned int kernel_size = kernel_geometry.w;

    switch (kernel_size)
    {
                  case 1:  cuda_convolution_forward_kernel<1><<<grid, block>>>( output,
                                                                                input,
                                                                                weights,
                                                                                bias,

                                                                                output_geometry,
                                                                                input_geometry,
                                                                                kernel_geometry);
                              break;

                  case 3:  cuda_convolution_forward_kernel<3><<<grid, block>>>( output,
                                                                                input,
                                                                                weights,
                                                                                bias,

                                                                                output_geometry,
                                                                                input_geometry,
                                                                                kernel_geometry);
                              break;


                  default:
                          cuda_convolution_forward_kernel_any_size<<<grid, block>>>(
                                                                                    output,
                                                                                    input,
                                                                                    weights,
                                                                                    bias,

                                                                                    output_geometry,
                                                                                    input_geometry,
                                                                                    kernel_geometry);
                            break;
    }

    cudaDeviceSynchronize();
}
