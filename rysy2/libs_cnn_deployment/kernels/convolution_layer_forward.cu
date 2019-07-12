#include <cuda_fp16.h>

#include "convolution_layer_forward.cuh"

#include "../cuda_float_allocator.cuh"

#define TILE_SIZE           ((unsigned int)16)
#define KERNEL_MAX_SIZE     ((unsigned int)3)
#define SHARED_SIZE         (TILE_SIZE + KERNEL_MAX_SIZE - 1)

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

    __shared__ float w_shared[kernel_size][kernel_size];

    if ((filter < output_shape.d) && (y < input_size_y) && (x < input_size_x))
    {
        float sum = bias[filter];

        for (unsigned int ch = 0; ch < input_shape.d; ch++)
        {
            unsigned int offset = filter*kernel_size*kernel_size*input_shape.d;
            if ( (threadIdx.x < kernel_size) && (threadIdx.y < kernel_size) )
            {
                unsigned int w_ofs = kernel_size*kernel_size*ch + offset;
                w_shared[threadIdx.y][threadIdx.x] = w[w_ofs + threadIdx.y*kernel_size + threadIdx.x];
            }

            __syncthreads();

            unsigned int input_idx  = (ch*input_shape.h + y)*input_shape.w + x;

            if (kernel_size == 1)
            {
                sum+= w_shared[0][0]*input[input_idx];
                input_idx+= input_shape.w;
            }

            if (kernel_size == 3)
            {
                sum+= w_shared[0][0]*input[input_idx++];
                sum+= w_shared[0][1]*input[input_idx++];
                sum+= w_shared[0][2]*input[input_idx++];
                input_idx+= input_shape.w - kernel_size;

                sum+= w_shared[1][0]*input[input_idx++];
                sum+= w_shared[1][1]*input[input_idx++];
                sum+= w_shared[1][2]*input[input_idx++];
                input_idx+= input_shape.w - kernel_size;

                sum+= w_shared[2][0]*input[input_idx++];
                sum+= w_shared[2][1]*input[input_idx++];
                sum+= w_shared[2][2]*input[input_idx++];
                input_idx+= input_shape.w - kernel_size;
            }

            __syncthreads();
        }

        //ReLU
        if (sum < 0.0)
            sum = 0.0;

        unsigned int output_idx = (filter*input_shape.h + y + k_half)*input_shape.w + x + k_half;
        output[output_idx] = sum;
    }
}
*/

/*
__float2half(x)
__half2float(x)
__hadd(a, b)
__hmul(a, b)
*/


#define get_input_idx(ch, y, x) ((ch*input_shape.h + y)*input_shape.w + x)

template<unsigned int kernel_size>
__global__
void cuda_convolution_forward_kernel(   float *output,
                                        const float *input,
                                        const float *w,
                                        const float *bias,

                                        const sShape output_shape,
                                        const sShape input_shape,
                                        const sShape kernel_shape,

                                        const unsigned int input_size_x,
                                        const unsigned int input_size_y
                                    )
{
    unsigned int x      = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y      = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int filter = threadIdx.z + blockIdx.z*blockDim.z;

    unsigned int k_half = (kernel_size - 1)/2;

    if (filter < output_shape.d)
    if (y < input_size_y)
    if (x < input_size_x)
    {
        __shared__ float w_shared[kernel_size][kernel_size];
        __shared__ float input_shared[SHARED_SIZE][SHARED_SIZE];

        float activation = bias[filter];

        for (unsigned int ch = 0; ch < input_shape.d; ch++)
        {
            unsigned int offset = filter*kernel_size*kernel_size*input_shape.d;
            if ( (threadIdx.x < kernel_size) && (threadIdx.y < kernel_size) )
            {
                unsigned int w_ofs = kernel_size*kernel_size*ch + offset;
                w_shared[threadIdx.y][threadIdx.x] = w[w_ofs + threadIdx.y*kernel_size + threadIdx.x];
            }

            unsigned int idx = get_input_idx(ch, y + 1, x + 0);

            input_shared[threadIdx.y + 1][threadIdx.x + 0] = input[idx]; idx++;
            input_shared[threadIdx.y + 1][threadIdx.x + 1] = input[idx]; idx++;
            input_shared[threadIdx.y + 1][threadIdx.x + 2] = input[idx];

            if (threadIdx.y == 0)
            {
                unsigned int idx = get_input_idx(ch, y + 0, x + 0);
                input_shared[0][threadIdx.x + 0] = input[idx]; idx++;
                input_shared[0][threadIdx.x + 1] = input[idx]; idx++;
                input_shared[0][threadIdx.x + 2] = input[idx];
            }
            else
            if (threadIdx.y == TILE_SIZE-1)
            {
                unsigned int idx = get_input_idx(ch, y + 2, x + 0);

                input_shared[TILE_SIZE-1 + 2][threadIdx.x + 0] = input[idx]; idx++;
                input_shared[TILE_SIZE-1 + 2][threadIdx.x + 1] = input[idx]; idx++;
                input_shared[TILE_SIZE-1 + 2][threadIdx.x + 2] = input[idx];
            }



            __syncthreads();


            if (kernel_size == 1)
            {
                activation+= w_shared[0][0] * input_shared[threadIdx.y][threadIdx.x];
            }

            if (kernel_size == 3)
            {
                activation+= w_shared[0][0] * input_shared[threadIdx.y + 0][threadIdx.x + 0];
                activation+= w_shared[0][1] * input_shared[threadIdx.y + 0][threadIdx.x + 1];
                activation+= w_shared[0][2] * input_shared[threadIdx.y + 0][threadIdx.x + 2];

                activation+= w_shared[1][0] * input_shared[threadIdx.y + 1][threadIdx.x + 0];
                activation+= w_shared[1][1] * input_shared[threadIdx.y + 1][threadIdx.x + 1];
                activation+= w_shared[1][2] * input_shared[threadIdx.y + 1][threadIdx.x + 2];

                activation+= w_shared[2][0] * input_shared[threadIdx.y + 2][threadIdx.x + 0];
                activation+= w_shared[2][1] * input_shared[threadIdx.y + 2][threadIdx.x + 1];
                activation+= w_shared[2][2] * input_shared[threadIdx.y + 2][threadIdx.x + 2];
            }

            __syncthreads();
        }

        //ReLU
        if (activation < 0.0)
            activation = 0.0;

        unsigned int output_idx = (filter*input_shape.h + y + k_half)*input_shape.w + x + k_half;
        output[output_idx] = activation;
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


void convolution_layer_forward( float *output, float *input,
                                float *weights, float *bias,
                                sShape input_shape,
                                sShape kernel_shape,
                                sShape output_shape )
{
    unsigned int input_size_y = input_shape.h - kernel_shape.h;
    unsigned int input_size_x = input_shape.w - kernel_shape.w;

    kernel_shape.d = output_shape.d;
    unsigned int kernel_size = kernel_shape.w;


    if ((kernel_size == 1) || (kernel_size == 3))
    {
        unsigned int input_size_x = input_shape.w - 2*((kernel_size - 1)/2);
        unsigned int input_size_y = input_shape.h - 2*((kernel_size - 1)/2);

        dim3 block(TILE_SIZE, TILE_SIZE, 1);
        dim3 grid(  (input_size_x      + block.x)/block.x,
                    (input_size_y      + block.y)/block.y,
                    (output_shape.d + block.z)/block.z );


        if (kernel_size == 1)
            cuda_convolution_forward_kernel<1><<<grid, block>>>( output,
                                                                 input,
                                                                 weights,
                                                                 bias,
                                                                 output_shape,
                                                                 input_shape,
                                                                 kernel_shape,
                                                                 input_size_x,
                                                                 input_size_y);
        if (kernel_size == 3)
            cuda_convolution_forward_kernel<3><<<grid, block>>>( output,
                                                                 input,
                                                                 weights,
                                                                 bias,
                                                                 output_shape,
                                                                 input_shape,
                                                                 kernel_shape,
                                                                 input_size_x,
                                                                 input_size_y);
    }
    else
    {
        dim3 block(8, 8, 1);
        dim3 grid(  (input_size_x      + block.x + 1)/block.x,
                    (input_size_y      + block.y + 1)/block.y,
                    (kernel_shape.d + block.z + 1)/block.z );

        cuda_convolution_forward_kernel_any_size<<<grid, block>>>(
                                                                  output,
                                                                  input,
                                                                  weights,
                                                                  bias,

                                                                  output_shape,
                                                                  input_shape,
                                                                  kernel_shape);
    }



    cudaDeviceSynchronize();
}
