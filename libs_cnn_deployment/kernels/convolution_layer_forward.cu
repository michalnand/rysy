#include "convolution_layer_forward.cuh"

#include "../cuda_float_allocator.cuh"

/*
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
*/


#define TILE_SIZE           ((unsigned int)32)
#define KERNEL_MAX_SIZE     ((unsigned int)3)

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

    if ((filter < output_geometry.d) && (y < input_size_y) && (x < input_size_x))
    {
        __shared__ float w_shared[kernel_size][kernel_size];

        float sum = bias[filter];
        unsigned int w_ofs = filter*kernel_size*kernel_size*input_geometry.d;

        for (unsigned int ch = 0; ch < input_geometry.d; ch++)
        {
            __syncthreads();

            if ( (threadIdx.x < kernel_size) && (threadIdx.y < kernel_size) )
            {
                w_shared[threadIdx.y][threadIdx.x] = w[w_ofs + threadIdx.y*kernel_size + threadIdx.x];
            }
            w_ofs+= kernel_size*kernel_size;

            __syncthreads();

            unsigned int input_idx  = (ch*input_geometry.h + y)*input_geometry.w + x;

            if (kernel_size == 1)
            {
                sum+= w_shared[0][0]*input[input_idx];
                input_idx+= input_geometry.w;
            }

            if (kernel_size == 3)
            {
                sum+= w_shared[0][0]*input[input_idx++];
                sum+= w_shared[0][1]*input[input_idx++];
                sum+= w_shared[0][2]*input[input_idx++];
                input_idx+= input_geometry.w - kernel_size;

                sum+= w_shared[1][0]*input[input_idx++];
                sum+= w_shared[1][1]*input[input_idx++];
                sum+= w_shared[1][2]*input[input_idx++];
                input_idx+= input_geometry.w - kernel_size;

                sum+= w_shared[2][0]*input[input_idx++];
                sum+= w_shared[2][1]*input[input_idx++];
                sum+= w_shared[2][2]*input[input_idx++];
                input_idx+= input_geometry.w - kernel_size;
            }
        }

        //ReLU
        if (sum < 0.0)
            sum = 0.0;

        unsigned int output_idx = (filter*input_geometry.h + y + k_half)*input_geometry.w + x + k_half;
        output[output_idx] = sum;
    }
}


/*
#define TILE_SIZE           ((unsigned int)32)
#define KERNEL_MAX_SIZE     ((unsigned int)3)

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


    __shared__ float w_shared[kernel_size][kernel_size];
    __shared__ float input_shared[TILE_SIZE + KERNEL_MAX_SIZE][TILE_SIZE + KERNEL_MAX_SIZE];

    if (filter < output_geometry.d)
    if (y < input_size_y)
    if (x < input_size_x)
    {
        float sum = bias[filter];
        unsigned int w_ofs = filter*kernel_size*kernel_size*input_geometry.d;

        for (unsigned int ch = 0; ch < input_geometry.d; ch++)
        {
            __syncthreads();

            unsigned int input_offset = ch*input_geometry.h*input_geometry.w;

            if ( (threadIdx.x < kernel_size) && (threadIdx.y < kernel_size) )
            {
                w_shared[threadIdx.y][threadIdx.x] = w[w_ofs + threadIdx.y*kernel_size + threadIdx.x];
            }
            w_ofs+= kernel_size*kernel_size;



            if ( (threadIdx.x < TILE_SIZE) && (threadIdx.y < TILE_SIZE) )
            {
                unsigned int input_idx;
                input_idx  = input_offset + y*input_geometry.w + x;
                input_shared[threadIdx.y][threadIdx.x] = input[input_idx];
            }
            __syncthreads();


            if (kernel_size == 1)
            {
                sum+= w_shared[0][0]*input_shared[threadIdx.y + 0][threadIdx.x + 0];
            }

            if (kernel_size == 3)
            {
                sum+= w_shared[0][0]*input_shared[threadIdx.y + 0][threadIdx.x + 0];
                sum+= w_shared[0][1]*input_shared[threadIdx.y + 0][threadIdx.x + 1];
                sum+= w_shared[0][2]*input_shared[threadIdx.y + 0][threadIdx.x + 2];

                sum+= w_shared[1][0]*input_shared[threadIdx.y + 1][threadIdx.x + 0];
                sum+= w_shared[1][1]*input_shared[threadIdx.y + 1][threadIdx.x + 1];
                sum+= w_shared[1][2]*input_shared[threadIdx.y + 1][threadIdx.x + 2];

                sum+= w_shared[2][0]*input_shared[threadIdx.y + 2][threadIdx.x + 0];
                sum+= w_shared[2][1]*input_shared[threadIdx.y + 2][threadIdx.x + 1];
                sum+= w_shared[2][2]*input_shared[threadIdx.y + 2][threadIdx.x + 2];
            }

        }


        //apply ReLU
        if (sum < 0.0)
            sum = 0.0;

        unsigned int output_idx = (filter*input_geometry.h + y + k_half)*input_geometry.w + x + k_half;
        output[output_idx] = sum;
    }
}
*/

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
    unsigned int kernel_size = kernel_geometry.w;


    if ((kernel_size == 1) || (kernel_size == 3))
    {
        dim3 block(TILE_SIZE, TILE_SIZE, 1);
        dim3 grid(  (input_size_x      + block.x + 1)/block.x,
                    (input_size_y      + block.y + 1)/block.y,
                    (output_geometry.d + block.z + 1)/block.z );

        if (kernel_size == 1)
            cuda_convolution_forward_kernel<1><<<grid, block>>>( output,
                                                                 input,
                                                                 weights,
                                                                 bias,
                                                                 output_geometry,
                                                                 input_geometry,
                                                                 kernel_geometry);
        if (kernel_size == 3)
            cuda_convolution_forward_kernel<3><<<grid, block>>>( output,
                                                                 input,
                                                                 weights,
                                                                 bias,
                                                                 output_geometry,
                                                                 input_geometry,
                                                                 kernel_geometry);
    }
    else
    {
        dim3 block(16, 16, 1);
        dim3 grid(  (input_size_x      + block.x + 1)/block.x,
                    (input_size_y      + block.y + 1)/block.y,
                    (kernel_geometry.d + block.z + 1)/block.z );

        cuda_convolution_forward_kernel_any_size<<<grid, block>>>(
                                                                  output,
                                                                  input,
                                                                  weights,
                                                                  bias,

                                                                  output_geometry,
                                                                  input_geometry,
                                                                  kernel_geometry);
    }



    cudaDeviceSynchronize();
}
