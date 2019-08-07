#include "gru_gate.cuh"

#define SIGMOID(x)    (  1.0 / (1.0 + exp(-x)))
#define TANH(x)        (tanh(x))
#define D_SIGMOID(y)   (y*(1.0 - y))
#define D_TANH(y)    (1.0 - y*y)


__host__
void cpu_gru_gate_forward_kernel(   float *h_next,

                                    float *control,
                                    float *h,
                                    float *update,

                                    unsigned int output_size)
{
    for (unsigned int idx = 0; idx < output_size; idx++)
    {
        float c = SIGMOID(control[idx]);
        float u = TANH(update[idx]);

        //process GRU gate output
        h_next[idx] = (1.0 - c)*h[idx] + c*u;
    }
}


__global__
void cuda_gru_gate_forward_kernel(  float *h_next,

                                    float *control,
                                    float *h,
                                    float *update,

                                    unsigned int output_size)
{
    unsigned int idx  = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < output_size)
    {
        float c = SIGMOID(control[idx]);
        float u = TANH(update[idx]);

        //process GRU gate output
        h_next[idx] = (1.0 - c)*h[idx] + c*u;
    }
}



__host__
void cpu_gru_gate_backward_kernel(      float *h_next,
                                        float *control,
                                        float *h,
                                        float *update,

                                        float *error,

                                        float *h_error_back,
                                        float *control_error_back,
                                        float *update_error_back,

                                        unsigned int output_size)
{
    for (unsigned int idx = 0; idx < output_size; idx++)
    {
        float err = error[idx];

        float c         = SIGMOID(control[idx]);
        float u         = TANH(update[idx]);
        float c_der     = D_SIGMOID(c);
        float u_der     = D_TANH(u);

        h_error_back[idx]       = (1.0 - c)*err;
        update_error_back[idx]  = (err*c)*u_der;
        control_error_back[idx] = (err*u - err*h[idx])*c_der;
    }
}


__global__
void cuda_gru_gate_backward_kernel( float *h_next,
                                    float *h,

                                    float *control,
                                    float *update,

                                    float *error,

                                    float *h_error_back,
                                    float *control_error_back,
                                    float *update_error_back,

                                    unsigned int output_size)
{
    unsigned int idx  = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < output_size)
    {
        float err = error[idx];

        float c         = SIGMOID(control[idx]);
        float u         = TANH(update[idx]);
        float c_der     = D_SIGMOID(c);
        float u_der     = D_TANH(u);

        h_error_back[idx]       = (1.0 - c)*err;
        update_error_back[idx]  = (err*c)*u_der;
        control_error_back[idx] = (err*u - err*h[idx])*c_der;
    }
}





void gru_gate_forward(  Tensor &h_next,

                        Tensor &control,
                        Tensor &h,
                        Tensor &update)
{
    unsigned int output_size  = h_next.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(32);
        dim3 grid((output_size  + block.x + 1)/block.x);

        cuda_gru_gate_forward_kernel<<<grid, block>>>(  h_next.v,

                                                        control.v,
                                                        h.v,
                                                        update.v,
                                                        output_size);

        cudaDeviceSynchronize();

    #else


        cpu_gru_gate_forward_kernel(    h_next.v,

                                        control.v,
                                        h.v,
                                        update.v,
                                        output_size);

    #endif
}

void gru_gate_backward( Tensor &h_next,

                        Tensor &control,
                        Tensor &h,
                        Tensor &update,

                        Tensor &error,

                        Tensor &control_error_back,
                        Tensor &h_error_back,
                        Tensor &update_error_back)
{
    unsigned int output_size  = h_next.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(32);
        dim3 grid((output_size  + block.x + 1)/block.x);

        cuda_gru_gate_backward_kernel<<<grid, block>>>(     h_next.v,

                                                            control.v,
                                                            h.v,
                                                            update.v,

                                                            error.v,

                                                            control_error_back.v,
                                                            h_error_back.v,
                                                            update_error_back.v,

                                                            output_size);

        cudaDeviceSynchronize();

    #else


        cpu_gru_gate_backward_kernel(   h_next.v,

                                        control.v,
                                        h.v,
                                        update.v,

                                        error.v,
                                        
                                        control_error_back.v,
                                        h_error_back.v,
                                        update_error_back.v,

                                        output_size);

    #endif
}
