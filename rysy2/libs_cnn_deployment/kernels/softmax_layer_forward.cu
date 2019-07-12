#include "convolution_layer_forward.cuh"

#include "../cuda_float_allocator.cuh"

__global__
void cuda_softmax_forward_kernel(  float *output,
                                   float *input,
                                   sShape input_shape)
{
    unsigned int x      = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y      = threadIdx.y + blockIdx.y*blockDim.y;

    if (y < input_shape.h)
    if (x < input_shape.w)
    {
        unsigned int input_size = input_shape.w*input_shape.h;
        unsigned int input_idx;

        float max = -10000000.0;

        input_idx  = y*input_shape.w + x;
        for (unsigned int filter = 0; filter < input_shape.d; filter++)
        {
            if (input[input_idx] > max)
                max = input[input_idx];

            input_idx+= input_size;
        }

        input_idx  = y*input_shape.w + x;
        float sum = 0.00000001;
        for (unsigned int filter = 0; filter < input_shape.d; filter++)
        {
            sum+= exp(input[input_idx] - max);
            input_idx+= input_size;
        }

        input_idx  = y*input_shape.w + x;
        for (unsigned int filter = 0; filter < input_shape.d; filter++)
        {
            output[input_idx] = exp(input[input_idx] - max)/sum;
            input_idx+= input_size;
        }
    }
}


void softmax_layer_forward( float *output, float *input,
                            sShape input_shape)
{
    unsigned int input_size_y = input_shape.h;
    unsigned int input_size_x = input_shape.w;

    dim3 block(8, 8);
    dim3 grid((input_size_x  + block.x + 1)/block.x,
              (input_size_y  + block.y + 1)/block.y);

    cuda_softmax_forward_kernel<<<grid, block>>>(   output,
                                                    input,
                                                    input_shape);
    cudaDeviceSynchronize();
}
