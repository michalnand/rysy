#include "rl_tanh_block_layer.cuh"


#define RL_BLOCK_ACTIVATION(x)                  tanh(x)
#define RL_BLOCK_ACTIVATION_DERIVATIVE(x)       (1.0 - x*x)

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

        output[neuron_idx] = RL_BLOCK_ACTIVATION(sum);
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

        output[neuron_idx] = RL_BLOCK_ACTIVATION(sum);
    }
}


void rl_tanh_block_layer_forward(   Tensor &output, Tensor &inputx, Tensor &inputh,
                                    Tensor &wx, Tensor &wh, Tensor &bias)
{
    unsigned int input_x_size = inputx.size();
    unsigned int input_h_size = inputh.size();
    unsigned int output_size  = output.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(16);
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
void cpu_rl_tanh_block_backward_kernel(    float *error_back,
                                            float *output,
                                            float *error,

                                            float *weights,

                                            unsigned int input_size,
                                            unsigned int output_size )
{
    for (unsigned int input_idx = 0; input_idx < input_size; input_idx++)
    {
        unsigned int w_ofs = 0;
        float sum = 0.0;
        for (unsigned int i = 0; i < output_size; i++)
        {
            float err = error[i]*RL_BLOCK_ACTIVATION_DERIVATIVE(output[i]);
            sum+= weights[w_ofs + input_idx]*err;
            w_ofs+= input_size;
        }

        error_back[input_idx] = sum;
    }
}


__global__
void cuda_rl_tanh_block_backward_kernel(    float *error_back,
                                            float *output,
                                            float *error,

                                            float *weights,

                                            unsigned int input_size,
                                            unsigned int output_size )
{
    unsigned int input_idx        = threadIdx.x + blockIdx.x*blockDim.x;

    if (input_idx < input_size)
    {
        unsigned int w_ofs = 0;
        float sum = 0.0;
        for (unsigned int i = 0; i < output_size; i++)
        {
            float err = error[i]*RL_BLOCK_ACTIVATION_DERIVATIVE(output[i]);
            sum+= weights[w_ofs + input_idx]*err;
            w_ofs+= input_size;
        }

        error_back[input_idx] = sum;
    }
}



void rl_tanh_block_layer_backward(Tensor &error_back_x, Tensor &error_back_h, Tensor &inputx, Tensor &inputh, Tensor &output, Tensor &error, Tensor &wx, Tensor &wh)
{
    unsigned int input_x_size = inputx.size();
    unsigned int input_h_size = inputh.size();
    unsigned int output_size  = output.size();

    #ifdef NETWORK_USE_CUDA

        {
            dim3 block(16);
            dim3 grid((input_x_size  + block.x + 1)/block.x);


            cuda_rl_tanh_block_backward_kernel<<<grid, block>>> (   error_back_x.v,
                                                                    output.v,
                                                                    error.v,
                                                                    wx.v,

                                                                    input_x_size,
                                                                    output_size);


            cudaDeviceSynchronize();
        }
        {
            dim3 block(16);
            dim3 grid((input_h_size  + block.x + 1)/block.x);


            cuda_rl_tanh_block_backward_kernel<<<grid, block>>> (   error_back_h.v,
                                                                    output.v,
                                                                    error.v,
                                                                    wh.v,

                                                                    input_h_size,
                                                                    output_size);


            cudaDeviceSynchronize();
        }

    #else
        cpu_rl_tanh_block_backward_kernel( error_back_x.v,
                                           output.v,
                                           error.v,
                                           wx.v,

                                           input_x_size,
                                           output_size);

        cpu_rl_tanh_block_backward_kernel( error_back_h.v,
                                           output.v,
                                           error.v,
                                           wh.v,

                                           input_h_size,
                                           output_size);

    #endif
}



__host__
void cpu_rl_tanh_block_gradient_kernel(    float *w_grad,
                                            float *input,
                                            float *output,
                                            float *error,

                                            unsigned int input_size,
                                            unsigned int output_size)
{
    for (unsigned int output_idx = 0; output_idx < output_size; output_idx++)
    {
        unsigned int w_ofs = output_idx*input_size;
        float err = error[output_idx]*RL_BLOCK_ACTIVATION_DERIVATIVE(output[output_idx]);

        for (unsigned int i = 0; i < input_size; i++)
            w_grad[w_ofs + i]+= err*input[i];
    }
}

__global__
void cuda_rl_tanh_block_gradient_kernel(    float *w_grad,
                                            float *input,
                                            float *output,
                                            float *error,

                                            unsigned int input_size,
                                            unsigned int output_size)
{
    unsigned int output_idx        = threadIdx.x + blockIdx.x*blockDim.x;

    if (output_idx < output_size)
    {
        unsigned int w_ofs = output_idx*input_size;
        float err = error[output_idx]*RL_BLOCK_ACTIVATION_DERIVATIVE(output[output_idx]);

        for (unsigned int i = 0; i < input_size; i++)
            w_grad[w_ofs + i]+= err*input[i];
    }
}


void rl_tanh_block_layer_gradient(Tensor &wx_grad, Tensor &wh_grad, Tensor &inputx, Tensor &inputh, Tensor &output, Tensor &error, Tensor &error_h)
{
    unsigned int input_x_size = inputx.size();
    unsigned int input_h_size = inputh.size();
    unsigned int output_size  = output.size();


    #ifdef NETWORK_USE_CUDA

        dim3 block(16);
        dim3 grid((output_size  + block.x + 1)/block.x);

        cuda_rl_tanh_block_gradient_kernel<<<grid, block>>>( wx_grad.v,
                                                             inputx.v,
                                                             output.v,
                                                             error.v,

                                                             input_x_size,
                                                             output_size );
        cudaDeviceSynchronize();


        cuda_rl_tanh_block_gradient_kernel<<<grid, block>>>( wh_grad.v,
                                                             inputh.v,
                                                             output.v,
                                                             error.v,

                                                             input_h_size,
                                                             output_size );
        cudaDeviceSynchronize();

    #else


        cpu_rl_tanh_block_gradient_kernel(                   wx_grad.v,
                                                             inputx.v,
                                                             output.v,
                                                             error.v,

                                                             input_x_size,
                                                             output_size );


        cpu_rl_tanh_block_gradient_kernel(                   wh_grad.v,
                                                             inputh.v,
                                                             output.v,
                                                             error.v,

                                                             input_h_size,
                                                             output_size );



    #endif
}


__host__
void cpu_rl_tanh_block_update_bias_kernel(float *bias, float *output, float *error, float learning_rate, unsigned int output_size)
{
    for (unsigned int output_idx = 0; output_idx < output_size; output_idx++)
    {
        float err = error[output_idx]*RL_BLOCK_ACTIVATION_DERIVATIVE(output[output_idx]);

        bias[output_idx]+= learning_rate*err;
    }
}

__global__
void cuda_rl_tanh_block_update_bias_kernel(float *bias, float *output, float *error, float learning_rate, unsigned int output_size)
{
    unsigned int output_idx        = threadIdx.x + blockIdx.x*blockDim.x;

    if (output_idx < output_size)
    {
        float err = error[output_idx]*RL_BLOCK_ACTIVATION_DERIVATIVE(output[output_idx]);

        bias[output_idx]+= learning_rate*err;
    }
}

void rl_tanh_block_layer_update_bias(Tensor &bias, Tensor &output, Tensor &error, float learning_rate)
{
    unsigned int output_size  = output.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(16);
        dim3 grid((output_size  + block.x + 1)/block.x);

        cuda_rl_tanh_block_update_bias_kernel<<<grid, block>>>( bias.v,
                                                                output.v,
                                                                error.v,

                                                                learning_rate,
                                                                output_size );
        cudaDeviceSynchronize();

    #else

        cpu_rl_tanh_block_update_bias_kernel(                   bias.v,
                                                                output.v,
                                                                error.v,

                                                                learning_rate,
                                                                output_size );

    #endif
}
