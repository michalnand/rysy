#include "lstm_gate.cuh"

#define SIGMOID(x)          (1.0 / (1.0 + exp(-x)))
#define TANH(x)             (tanh(x))
#define SIGMOID_DER(x)      (SIGMOID(x)*(1.0 - SIGMOID(x)))
#define TANH_DER(x)         (1.0 - tanh(x)*tanh(x))

__host__
void cpu_lstm_gate_forward_kernel(  float *c_,
                                    float *forget_,
                                    float *input_,
                                    float *input_candidate_,

                                    float *output_candidate_,

                                    float *c_next_,
                                    float *h_next_,

                                    unsigned int output_size)
{
    for (unsigned int idx = 0; idx < output_size; idx++)
    {
        float c                 = c_[idx];
        float output_candidate  = output_candidate_[idx];

        float forget            = SIGMOID(forget_[idx]);
        float input             = SIGMOID(input_[idx]);
        float input_candidate   = TANH(input_candidate_[idx]);


        float c_next = c*forget + input*input_candidate;
        float h_next = TANH(c_next)*output_candidate;

        c_next_[idx] = c_next;
        h_next_[idx] = h_next;
    }
}


__global__
void cuda_lstm_gate_forward_kernel(     float *c_,
                                        float *forget_,
                                        float *input_,
                                        float *input_candidate_,

                                        float *output_candidate_,

                                        float *c_next_,
                                        float *h_next_,

                                        unsigned int output_size)
{
    unsigned int idx  = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < output_size)
    {
        float c                 = c_[idx];
        float output_candidate  = output_candidate_[idx];

        float forget            = SIGMOID(forget_[idx]);
        float input             = SIGMOID(input_[idx]);
        float input_candidate   = TANH(input_candidate_[idx]);


        float c_next = c*forget + input*input_candidate;
        float h_next = TANH(c_next)*output_candidate;

        c_next_[idx] = c_next;
        h_next_[idx] = h_next;
    }
}


__host__
void cpu_lstm_gate_backward_kernel(     float *c_,
                                        float *forget_,
                                        float *input_,
                                        float *input_candidate_,
                                        float *output_candidate_,

                                        float *error_c_next,
                                        float *error_h_next,

                                        float *error_back_c_,
                                        float *error_back_forget_,
                                        float *error_back_input_,
                                        float *error_back_input_candidate_,
                                        float *error_back_output_candidate_,

                                        unsigned int output_size)
{
    for (unsigned int idx = 0; idx < output_size; idx++)
    {
        float c                 = c_[idx];
        float output_candidate  = output_candidate_[idx];

        float forget            = SIGMOID(forget_[idx]);
        float input             = SIGMOID(input_[idx]);
        float input_candidate   = TANH(input_candidate_[idx]);


        float c_next = c*forget + input*input_candidate;
        float h_next = TANH(c_next)*output_candidate;


        float error_back_output_candidate = TANH(c_next)*error_h_next[idx] + TANH_DER(c_next)*error_c_next[idx]*h_next;
        error_back_output_candidate_[idx] = error_back_output_candidate;


        float ce_tmp = error_c_next[idx] + TANH_DER(c_next)*(error_h_next[idx]*output_candidate);

        error_back_c_[idx]                  = ce_tmp*forget;
        error_back_forget_[idx]             = SIGMOID_DER(forget_[idx])*c;
        error_back_input_[idx]              = ce_tmp*SIGMOID_DER(input_[idx])*TANH(input_candidate_[idx]);
        error_back_input_candidate_[idx]    = ce_tmp*SIGMOID(input_[idx])*TANH_DER(input_candidate_[idx]);
    }
}


__global__
void cuda_lstm_gate_backward_kernel(    float *c_,
                                        float *forget_,
                                        float *input_,
                                        float *input_candidate_,
                                        float *output_candidate_,

                                        float *error_c_next,
                                        float *error_h_next,

                                        float *error_back_c_,
                                        float *error_back_forget_,
                                        float *error_back_input_,
                                        float *error_back_input_candidate_,
                                        float *error_back_output_candidate_,

                                        unsigned int output_size)
{
    unsigned int idx  = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx < output_size)
    {
        float c                 = c_[idx];
        float output_candidate  = output_candidate_[idx];

        float forget            = SIGMOID(forget_[idx]);
        float input             = SIGMOID(input_[idx]);
        float input_candidate   = TANH(input_candidate_[idx]);


        float c_next = c*forget + input*input_candidate;
        float h_next = TANH(c_next)*output_candidate;


        float error_back_output_candidate = TANH(c_next)*error_h_next[idx] + TANH_DER(c_next)*error_c_next[idx]*h_next;
        error_back_output_candidate_[idx] = error_back_output_candidate;


        float ce_tmp = error_c_next[idx] + TANH_DER(c_next)*(error_h_next[idx]*output_candidate);

        error_back_c_[idx]                  = ce_tmp*forget;
        error_back_forget_[idx]             = SIGMOID_DER(forget_[idx])*c;
        error_back_input_[idx]              = ce_tmp*SIGMOID_DER(input_[idx])*TANH(input_candidate_[idx]);
        error_back_input_candidate_[idx]    = ce_tmp*SIGMOID(input_[idx])*TANH_DER(input_candidate_[idx]);
    }
}




void lstm_gate_forward(     Tensor &c,
                            Tensor &forget,
                            Tensor &input,
                            Tensor &input_candidate,

                            Tensor &output_candidate,

                            Tensor &c_next,
                            Tensor &h_next)
{
    unsigned int output_size  = c.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(32);
        dim3 grid((output_size  + block.x + 1)/block.x);

        cuda_lstm_gate_forward_kernel<<<grid, block>>>(     c.v,
                                                            forget.v,
                                                            input.v,
                                                            input_candidate.v,
                                                            output_candidate.v,
                                                            c_next.v,
                                                            h_next.v,
                                                            output_size);

        cudaDeviceSynchronize();

    #else


        cpu_lstm_gate_forward_kernel(                   c.v,
                                                        forget.v,
                                                        input.v,
                                                        input_candidate.v,
                                                        output_candidate.v,
                                                        c_next.v,
                                                        h_next.v,
                                                        output_size);

    #endif
}



void lstm_gate_backward(    Tensor &c,
                            Tensor &forget,
                            Tensor &input,
                            Tensor &input_candidate,
                            Tensor &output_candidate,

                            Tensor &error_c_next,
                            Tensor &error_h_next,

                            Tensor &error_back_c,
                            Tensor &error_back_forget,
                            Tensor &error_back_input,
                            Tensor &error_back_input_candidate,
                            Tensor &error_back_output_candidate)
{
    unsigned int output_size  = c.size();

    #ifdef NETWORK_USE_CUDA

        dim3 block(32);
        dim3 grid((output_size  + block.x + 1)/block.x);

        cuda_lstm_gate_backward_kernel<<<grid, block>>>(c.v,
                                                        forget.v,
                                                        input.v,
                                                        input_candidate.v,
                                                        output_candidate.v,

                                                        error_c_next.v,
                                                        error_h_next.v,

                                                        error_back_c.v,
                                                        error_back_forget.v,
                                                        error_back_input.v,
                                                        error_back_input_candidate.v,
                                                        error_back_output_candidate.v,

                                                        output_size);

        cudaDeviceSynchronize();

    #else


        cpu_lstm_gate_backward_kernel(                  c.v,
                                                        forget.v,
                                                        input.v,
                                                        input_candidate.v,
                                                        output_candidate.v,

                                                        error_c_next.v,
                                                        error_h_next.v,

                                                        error_back_c.v,
                                                        error_back_forget.v,
                                                        error_back_input.v,
                                                        error_back_input_candidate.v,
                                                        error_back_output_candidate.v,

                                                        output_size);

    #endif
}
