#include "gru_gate.cuh"

#define CONTROL_ACTIVATION(x)    (1.0 / (1.0 + exp(-x)))
#define UPDATE_ACTIVATION(x)    (tanh(x))
#define CONTROL_DER_ACTIVATION(x)   (CONTROL_ACTIVATION(x)*(1.0 - CONTROL_ACTIVATION(x)))
#define UPDATE_DER_ACTIVATION(x)    (1.0 - tanh(x)*tanh(x))


__host__
void cpu_gru_gate_forward_kernel(   float *h_next,
                                    float *h,
                                    float *x,

                                    float *control,
                                    float *update,

                                    unsigned int output_size)
{
    for (unsigned int idx = 0; idx < output_size; idx++)
    {
        float c = CONTROL_ACTIVATION(control[idx]);
        float u = UPDATE_ACTIVATION(update[idx]);

        //process GRU gate output
        h_next[idx] = (1.0 - c)*h[idx] + c*u;
    }
}


__global__
void cuda_gru_gate_forward_kernel(  float *h_next,
                                    float *h,
                                    float *x,

                                    float *control,
                                    float *update,

                                    unsigned int output_size)
{
    unsigned int idx  = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < output_size)
    {
        float c = CONTROL_ACTIVATION(control[idx]);
        float u = UPDATE_ACTIVATION(update[idx]);

        //process GRU gate output
        h_next[idx] = (1.0 - c)*h[idx] + c*u;
    }
}



__host__
void cpu_gru_gate_backward_kernel(      float *h_next,
                                        float *h,
                                        float *x,

                                        float *control,
                                        float *update,

                                        float *error,

                                        float *h_error_back,
                                        float *update_error_back,
                                        float *control_error_back,

                                        unsigned int output_size)
{
    for (unsigned int idx = 0; idx < output_size; idx++)
    {
        float err = error[idx];

        float c         = CONTROL_ACTIVATION(control[idx]);
        float u         = UPDATE_ACTIVATION(update[idx]);
        float c_der     = CONTROL_DER_ACTIVATION(control[idx]);
        float u_der     = UPDATE_DER_ACTIVATION(update[idx]);

        h_error_back[idx]       = (1.0 - c)*err;
        update_error_back[idx]  = (err*c)*u_der;
        control_error_back[idx] = (err*u - err*h[idx])*c_der;
    }
}


__global__
void cuda_gru_gate_backward_kernel( float *h_next,
                                    float *h,
                                    float *x,

                                    float *control,
                                    float *update,

                                    float *error,

                                    float *h_error_back,
                                    float *update_error_back,
                                    float *control_error_back,

                                    unsigned int output_size)
{
    unsigned int idx  = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < output_size)
    {
        float err = error[idx];

        float c         = CONTROL_ACTIVATION(control[idx]);
        float u         = UPDATE_ACTIVATION(update[idx]);
        float c_der     = CONTROL_DER_ACTIVATION(control[idx]);
        float u_der     = UPDATE_DER_ACTIVATION(update[idx]);

        h_error_back[idx]       = (1.0 - c)*err;
        update_error_back[idx]  = (err*c)*u_der;
        control_error_back[idx] = (err*u - err*h[idx])*c_der;
    }
}





void gru_gate_forward(  Tensor &h_next,
                        Tensor &h,
                        Tensor &x,

                        Tensor &control,
                        Tensor &update)
{
    unsigned int output_size  = h_next.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(32);
        dim3 grid((output_size  + block.x + 1)/block.x);

        cuda_gru_gate_forward_kernel<<<grid, block>>>(  h_next.v,
                                                        h.v,
                                                        x.v,
                                                        control.v,
                                                        update.v,
                                                        output_size);

        cudaDeviceSynchronize();

    #else


        cpu_gru_gate_forward_kernel(    h_next.v,
                                        h.v,
                                        x.v,
                                        control.v,
                                        update.v,
                                        output_size);

    #endif
}

void gru_gate_backward( Tensor &h_next,
                        Tensor &h,
                        Tensor &x,

                        Tensor &control,
                        Tensor &update,

                        Tensor &error,

                        Tensor &h_error_back,
                        Tensor &update_error_back,
                        Tensor &control_error_back)
{
    unsigned int output_size  = h_next.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(32);
        dim3 grid((output_size  + block.x + 1)/block.x);

        cuda_gru_gate_backward_kernel<<<grid, block>>>(     h_next.v,
                                                            h.v,
                                                            x.v,
                                                            control.v,
                                                            update.v,

                                                            error.v,

                                                            h_error_back.v,
                                                            update_error_back.v,
                                                            control_error_back.v,

                                                            output_size);

        cudaDeviceSynchronize();

    #else


        cpu_gru_gate_backward_kernel(   h_next.v,
                                        h.v,
                                        x.v,
                                        control.v,
                                        update.v,

                                        error.v,

                                        h_error_back.v,
                                        update_error_back.v,
                                        control_error_back.v,

                                        output_size);

    #endif
}
