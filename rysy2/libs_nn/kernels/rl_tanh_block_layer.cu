#include "rl_tanh_block_layer.cuh"

__host__
void cpu_rl_tanh_block_forward_kernel(      float *output,
                                            float *inputx,
                                            float *inputh,
                                            float *wx,
                                            float *wh,
                                            float *bias,

                                            unsigned int input_x_size,
                                            unsigned int input_h_size,
                                            unsigned int output_size )
{
    for (unsigned int neuron_idx = 0; neuron_idx < output_size; neuron_idx++)
    {
        unsigned int wx_ofs = neuron_idx*input_x_size;
        unsigned int wh_ofs = neuron_idx*input_h_size;

        float sum = bias[neuron_idx];

        for (unsigned int i = 0; i < input_x_size; i++)
            sum+= wx[i + wx_ofs]*inputx[i];

        for (unsigned int i = 0; i < input_h_size; i++)
            sum+= wh[i + wh_ofs]*inputh[i];

        output[neuron_idx] = tanh(sum);
    }
}

__global__
void cuda_rl_tanh_block_forward_kernel(     float *output,
                                            float *inputx,
                                            float *inputh,
                                            float *wx,
                                            float *wh,
                                            float *bias,

                                            unsigned int input_x_size,
                                            unsigned int input_h_size,
                                            unsigned int output_size )
{
    unsigned int neuron_idx        = threadIdx.x + blockIdx.x*blockDim.x;

    if (neuron_idx < output_size)
    {
        unsigned int wx_ofs = neuron_idx*input_x_size;
        unsigned int wh_ofs = neuron_idx*input_h_size;

        float sum = bias[neuron_idx];

        for (unsigned int i = 0; i < input_x_size; i++)
            sum+= wx[i + wx_ofs]*inputx[i];

        for (unsigned int i = 0; i < input_h_size; i++)
            sum+= wh[i + wh_ofs]*inputh[i];

        output[neuron_idx] = tanh(sum);
    }
}


void rl_tanh_block_layer_forward(   Tensor &output, Tensor &inputx, Tensor &inputh,
                                    Tensor &wx, Tensor &wh, Tensor &bias)
{
    unsigned int input_x_size = inputx.size();
    unsigned int input_h_size = inputh.size();
    unsigned int output_size  = output.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(32);
        dim3 grid((output_size  + block.x + 1)/block.x);

        cuda_rl_tanh_block_forward_kernel<<<grid, block>>>(   output.v,
                                                                inputx.v,
                                                                inputh.v,
                                                                wx.v,
                                                                wh.v,
                                                                bias.v,

                                                                input_x_size,
                                                                input_h_size,
                                                                output_size
                                                            );
        cudaDeviceSynchronize();

    #else

        cpu_rl_tanh_block_forward_kernel(                     output.v,
                                                                inputx.v,
                                                                inputh.v,
                                                                wx.v,
                                                                wh.v,
                                                                bias.v,

                                                                input_x_size,
                                                                input_h_size,
                                                                output_size);
    #endif
}


__host__
void cpu_rl_tanh_block_backward_kernel(     float *error_back_x,
                                            float *error_back_h,
                                            float *inputx,
                                            float *inputh,
                                            float *output,
                                            float *error,

                                            float *wx,
                                            float *wh,

                                            unsigned int input_x_size,
                                            unsigned int input_h_size,
                                            unsigned int output_size )
{
    for (unsigned int neuron_idx = 0; neuron_idx < output_size; neuron_idx++)
    {
        float error_der = (1.0 - output[neuron_idx]*output[neuron_idx])*error[neuron_idx];

        unsigned int wx_ofs = neuron_idx*input_x_size;
        unsigned int wh_ofs = neuron_idx*input_h_size;

        for (unsigned int i = 0; i < input_x_size; i++)
        {
            float error_back = wx[i + wx_ofs]*error_der;
            error_back_x[i]+= error_back;
        }

        for (unsigned int i = 0; i < input_h_size; i++)
        {
            float error_back = wh[i + wh_ofs]*error_der;
            error_back_h[i]+= error_back;
        }
    }
}


__global__
void cuda_rl_tanh_block_backward_kernel(    float *error_back_x,
                                            float *error_back_h,
                                            float *inputx,
                                            float *inputh,
                                            float *output,
                                            float *error,

                                            float *wx,
                                            float *wh,

                                            unsigned int input_x_size,
                                            unsigned int input_h_size,
                                            unsigned int output_size )
{
    unsigned int neuron_idx        = threadIdx.x + blockIdx.x*blockDim.x;

    if (neuron_idx < output_size)
    {
        float error_der = (1.0 - output[neuron_idx]*output[neuron_idx])*error[neuron_idx];

        unsigned int wx_ofs = neuron_idx*input_x_size;
        unsigned int wh_ofs = neuron_idx*input_h_size;

        for (unsigned int i = 0; i < input_x_size; i++)
        {
            float error_back = wx[i + wx_ofs]*error_der;
            atomicAdd(&error_back_x[i], error_back);
        }

        for (unsigned int i = 0; i < input_h_size; i++)
        {
            float error_back = wh[i + wh_ofs]*error_der;
            atomicAdd(&error_back_h[i], error_back);
        }
    }
}

void rl_tanh_block_layer_backward(Tensor &error_back_x, Tensor &error_back_h, Tensor &inputx, Tensor &inputh, Tensor &output, Tensor &error, Tensor &wx, Tensor &wh)
{
    unsigned int input_x_size = inputx.size();
    unsigned int input_h_size = inputh.size();
    unsigned int output_size  = output.size();

    error_back_x.clear();
    error_back_h.clear();

    #ifdef NETWORK_USE_CUDA

        dim3 block(32);
        dim3 grid((output_size  + block.x + 1)/block.x);

        cuda_rl_tanh_block_backward_kernel<<<grid, block>>>(    error_back_x.v,
                                                                error_back_h.v,
                                                                inputx.v,
                                                                inputh.v,
                                                                output.v,
                                                                error.v,
                                                                wx.v,
                                                                wh.v,

                                                                input_x_size,
                                                                input_h_size,
                                                                output_size
                                                             );
        cudaDeviceSynchronize();

    #else

        cpu_rl_tanh_block_backward_kernel(                      error_back_x.v,
                                                                error_back_h.v,
                                                                inputx.v,
                                                                inputh.v,
                                                                output.v,
                                                                error.v,
                                                                wx.v,
                                                                wh.v,

                                                                input_x_size,
                                                                input_h_size,
                                                                output_size
                                                            );
    #endif
}


__host__
void cpu_rl_tanh_block_gradient_kernel(     float *wx_grad,
                                            float *wh_grad,

                                            float *inputx,
                                            float *inputh,

                                            float *output,
                                            float *error,
                                            float *error_h,

                                            unsigned int input_x_size,
                                            unsigned int input_h_size,
                                            unsigned int output_size)
{
    for (unsigned int neuron_idx = 0; neuron_idx < output_size; neuron_idx++)
    {
        float error_der = (1.0 - output[neuron_idx]*output[neuron_idx])*(error[neuron_idx] + error_h[neuron_idx]);

        unsigned int wx_ofs = neuron_idx*input_x_size;
        unsigned int wh_ofs = neuron_idx*input_h_size;

        for (unsigned int i = 0; i < input_x_size; i++)
        {
            float w_grad = inputx[i]*error_der;
            wx_grad[i + wx_ofs] = w_grad;
        }

        for (unsigned int i = 0; i < input_h_size; i++)
        {
            float w_grad = inputh[i]*error_der;
            wh_grad[i + wh_ofs] = w_grad;
        }
    }
}

__global__
void cuda_rl_tanh_block_gradient_kernel(    float *wx_grad,
                                            float *wh_grad,

                                            float *inputx,
                                            float *inputh,

                                            float *output,
                                            float *error,
                                            float *error_h,

                                            unsigned int input_x_size,
                                            unsigned int input_h_size,
                                            unsigned int output_size)
{
    unsigned int neuron_idx        = threadIdx.x + blockIdx.x*blockDim.x;

    if (neuron_idx < output_size)
    {
        float error_der = (1.0 - output[neuron_idx]*output[neuron_idx])*(error[neuron_idx] + error_h[neuron_idx]);

        unsigned int wx_ofs = neuron_idx*input_x_size;
        unsigned int wh_ofs = neuron_idx*input_h_size;

        for (unsigned int i = 0; i < input_x_size; i++)
        {
            float w_grad = inputx[i]*error_der;
            atomicAdd(&wx_grad[i + wx_ofs], w_grad);
        }

        for (unsigned int i = 0; i < input_h_size; i++)
        {
            float w_grad = inputh[i]*error_der;
            atomicAdd(&wh_grad[i + wh_ofs], w_grad);
        }
    }
}

void rl_tanh_block_layer_gradient(Tensor &wx_grad, Tensor &wh_grad, Tensor &inputx, Tensor &inputh, Tensor &output, Tensor &error, Tensor &error_h)
{
    unsigned int input_x_size = inputx.size();
    unsigned int input_h_size = inputh.size();
    unsigned int output_size  = output.size();


    #ifdef NETWORK_USE_CUDA

        dim3 block(32);
        dim3 grid((output_size  + block.x + 1)/block.x);

        cuda_rl_tanh_block_gradient_kernel<<<grid, block>>>(    wx_grad.v,
                                                                wh_grad.v,

                                                                inputx.v,
                                                                inputh.v,

                                                                output.v,
                                                                error.v,
                                                                error_h.v,

                                                                input_x_size,
                                                                input_h_size,
                                                                output_size
                                                             );
        cudaDeviceSynchronize();

    #else

        cpu_rl_tanh_block_gradient_kernel(                      wx_grad.v,
                                                                wh_grad.v,

                                                                inputx.v,
                                                                inputh.v,

                                                                output.v,
                                                                error.v,
                                                                error_h.v,

                                                                input_x_size,
                                                                input_h_size,
                                                                output_size
                                                         );

    #endif
}
