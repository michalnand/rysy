#include "crop_layer.cuh"

__host__
void cpu_crop_forward_kernel(float *output, float *input,
                                unsigned int output_width,
                                unsigned int output_height,
                                unsigned int depth)
{
    unsigned int input_height = output_height + 2;
    unsigned int input_width  = output_width  + 2;

    unsigned int input_layer_size   = input_width*input_height;
    unsigned int output_layer_size  = output_width*output_height;


    for (unsigned int y = 0; y < output_height; y++)
    for (unsigned int x = 0; x < output_width; x++)
    {
        unsigned int input_idx  = (y + 1)*input_width  + (x + 1);
        unsigned int output_idx = y*output_width + x;

        for (unsigned int z = 0; z < depth; z++)
        {
            output[output_idx] = input[input_idx];
            input_idx+= input_layer_size;
            output_idx+= output_layer_size;
        }
    }
}

__global__
void cuda_crop_forward_kernel(float *output, float *input,
                                unsigned int output_width,
                                unsigned int output_height,
                                unsigned int depth)
{
    unsigned int x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (y < output_height)
    if (x < output_width)
    {
        unsigned int input_height = output_height + 2;
        unsigned int input_width  = output_width  + 2;

        unsigned int input_layer_size   = input_width*input_height;
        unsigned int output_layer_size  = output_width*output_height;

        unsigned int input_idx  = (y + 1)*input_width  + (x + 1);
        unsigned int output_idx = y*output_width + x;

        for (unsigned int z = 0; z < depth; z++)
        {
            output[output_idx] = input[input_idx];
            input_idx+= input_layer_size;
            output_idx+= output_layer_size;
        }
    }
}

void crop_layer_forward( Tensor &output, Tensor &input)
{
    unsigned int width  = output.shape().w();
    unsigned int height = output.shape().h();
    unsigned int depth  = output.shape().d();

    #ifdef NETWORK_USE_CUDA

        dim3 block(8, 8);
        dim3 grid((width + block.x + 1)/block.x, (height + block.y + 1)/block.y);

        cuda_crop_forward_kernel<<<grid, block>>>(output.v, input.v, width, height, depth);
        cudaDeviceSynchronize();

    #else

        cpu_crop_forward_kernel(output.v, input.v, width, height, depth);

    #endif
}



__host__
void cpu_crop_backward_kernel(  float *error_back, float *error,
                                unsigned int output_width,
                                unsigned int output_height,
                                unsigned int depth
                                )
{
    unsigned int input_height = output_height + 2;
    unsigned int input_width  = output_width  + 2;
    unsigned int input_layer_size   = input_width*input_height;
    unsigned int output_layer_size  = output_width*output_height;

    for (unsigned int y = 0; y < output_height; y++)
    for (unsigned int x = 0; x < output_width; x++)
    {
        unsigned int input_idx  = (y + 1)*input_width  + (x + 1);
        unsigned int output_idx = y*output_width + x;

        for (unsigned int z = 0; z < depth; z++)
        {
            error_back[input_idx] = error[output_idx];
            input_idx+= input_layer_size;
            output_idx+= output_layer_size;
        }
    }
}

__global__
void cuda_crop_backward_kernel( float *error_back, float *error,
                                unsigned int output_width,
                                unsigned int output_height,
                                unsigned int depth)
{
    unsigned int x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (y < output_height)
    if (x < output_width)
    {
        unsigned int input_height = output_height + 2;
        unsigned int input_width  = output_width  + 2;

        unsigned int input_layer_size   = input_width*input_height;
        unsigned int output_layer_size  = output_width*output_height;

        unsigned int input_idx  = (y + 1)*input_width  + (x + 1);
        unsigned int output_idx = y*output_width + x;

        for (unsigned int z = 0; z < depth; z++)
        {
            error_back[input_idx] = error[output_idx];
            input_idx+= input_layer_size;
            output_idx+= output_layer_size;
        }
    }
}

void crop_layer_backward(Tensor &error_back, Tensor &error)
{
    unsigned int width  = error.shape().w();
    unsigned int height = error.shape().h();
    unsigned int depth  = error.shape().d();

    #ifdef NETWORK_USE_CUDA

        dim3 block(8, 8);
        dim3 grid((width + block.x + 1)/block.x, (height + block.y + 1)/block.y);

        cuda_crop_backward_kernel<<<grid, block>>>(error_back.v, error.v, width, height, depth);
        cudaDeviceSynchronize();

    #else

        cpu_crop_backward_kernel(error_back.v, error.v, width, height, depth);

    #endif
}
