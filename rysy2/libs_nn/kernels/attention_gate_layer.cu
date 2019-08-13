#include "attention_gate_layer.cuh"

__host__
void cpu_attention_gate_forward_kernel(float *output, float *control, float *input, sShape shape)
{
    for (unsigned int ch = 0; ch < shape.d; ch++)
        for (unsigned int y = 0; y < shape.h; y++)
            for (unsigned int x = 0; x < shape.w; x++)
            {
                unsigned int idx, size = shape.w*shape.h*shape.d;


                float sum = 0.0;
                idx = (ch*shape.h + y)*shape.w + x;
                for (unsigned int t = 0; t < shape.t; t++)
                {
                    sum+= exp(control[idx]);
                    idx+= size;
                }

                idx = (ch*shape.h + y)*shape.w + x;
                float result = 0.0;
                for (unsigned int t = 0; t < shape.t; t++)
                {
                    result+= exp(control[idx])*input[idx];
                    idx+= size;
                }

                unsigned int out_idx = (ch*shape.h + y)*shape.w + x;
                output[out_idx] = result/sum;
            }
}

__global__
void cuda_attention_gate_forward_kernel(float *output, float *control, float *input, sShape shape)
{
    unsigned int x = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int ch = threadIdx.z + blockIdx.z*blockDim.z;


    if (ch < shape.d)
    if (y < shape.h)
    if (x < shape.w)
    {
        unsigned int idx, size = shape.w*shape.h*shape.d;


        float sum = 0.0;
        idx = (ch*shape.h + y)*shape.w + x;
        for (unsigned int t = 0; t < shape.t; t++)
        {
            sum+= exp(control[idx]);
            idx+= size;
        }

        idx = (ch*shape.h + y)*shape.w + x;
        float result = 0.0;
        for (unsigned int t = 0; t < shape.t; t++)
        {
            result+= exp(control[idx])*input[idx];
            idx+= size;
        }

        unsigned int out_idx = (ch*shape.h + y)*shape.w + x;
        output[out_idx] = result/sum;
    }
}

void attention_gate_layer_forward(Tensor &output, Tensor control, Tensor &input)
{
    unsigned int size = output.size();

    #ifdef NETWORK_USE_CUDA
        sShape shape = control.shape().get();

        dim3 block(4, 4, 4);
        dim3 grid((shape.w + block.x + 1)/block.x, (shape.h + block.y + 1)/block.y, (shape.d + block.z + 1)/block.z);

        cuda_attention_gate_forward_kernel<<<grid, block>>>(output.v, control.v, input.v, shape);
        cudaDeviceSynchronize();

    #else

        cpu_attention_gate_forward_kernel(output.v, control.v, input.v, shape);

    #endif
}


__host__
void cpu_attention_gate_backward_kernel(float *error_back_input, float *error_back_control, float *error, float *control, float *input, sShape shape)
{
    for (unsigned int ch = 0; ch < shape.d; ch++)
        for (unsigned int y = 0; y < shape.h; y++)
            for (unsigned int x = 0; x < shape.w; x++)
            {
                unsigned int idx, size = shape.w*shape.h*shape.d;


                float sum = 0.0;
                idx = (ch*shape.h + y)*shape.w + x;
                for (unsigned int t = 0; t < shape.t; t++)
                {
                    sum+= exp(control[idx]);
                    idx+= size;
                }

                float result = 0.0;
                idx = (ch*shape.h + y)*shape.w + x;
                for (unsigned int t = 0; t < shape.t; t++)
                {
                    result+= exp(control[idx])*input[idx];
                    idx+= size;
                }

                float output = result/sum;


                unsigned int out_idx = (ch*shape.h + y)*shape.w + x;

                idx = (ch*shape.h + y)*shape.w + x;
                for (unsigned int t = 0; t < shape.t; t++)
                {
                    error_back_input[idx] = output*input[idx];
                    idx+= size;
                }


            }
}

__global__
void cuda_attention_gate_backward_kernel(float *error_back, float *error, float *output, unsigned int size)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

}

void attention_gate_layer_backward( Tensor &error_back, Tensor &output, Tensor &error)
{
    unsigned int size = output.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(16);
        dim3 grid((size + block.x + 1)/block.x);

        //cuda_attention_gate_backward_kernel<<<grid, block>>>(error_back.v, error.v, output.v, size);
        //cudaDeviceSynchronize();

    #else

        //cpu_attention_gate_backward_kernel(error_back.v, error.v, output.v, size);

    #endif
}
