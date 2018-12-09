#include "gru_gate.cuh"

#define SIGMOID(x)            (1.0/(1.0 + exp(-x)))
#define SIGMOID_DERIVATIVE(x) (SIGMOID(x)*(1.0 - SIGMOID(x)))

__host__
void cpu_gru_gate_forward_kernel( float *output,
                                  float *state,
                                  float *state_update,
                                  float *gate,
                                  unsigned int size)
{
  for (unsigned int idx = 0; idx < size; idx++)
  {
    float su = SIGMOID(state_update[idx]);
    float g  = SIGMOID(gate[idx]);

    output[idx] = (1.0 - g)*state[idx] + g*su;
  }
}

__global__
void cuda_gru_gate_forward_kernel(  float *output,
                                    float *state,
                                    float *state_update,
                                    float *gate,
                                    unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    float su = SIGMOID(state_update[idx]);
    float g  = SIGMOID(gate[idx]);

    output[idx] = (1.0 - g)*state[idx] + g*su;
  }
}

void gru_gate_forward(Tensor &output, Tensor &state, Tensor &state_update, Tensor &gate)
{
  unsigned int size = output.size();

  #ifdef NETWORK_USE_CUDA
    unsigned int block_size = 16;
    if (size >= 256)
      block_size = 64;

    dim3 block(block_size);
    dim3 grid((size + block.x - 1)/block.x);

    cuda_gru_gate_forward_kernel<<<grid, block>>>(  output.v,

                                                    state.v,
                                                    state_update.v,
                                                    gate.v,

                                                    size);
    cudaDeviceSynchronize();
  #else
    cpu_gru_gate_forward_kernel(  output.v,

                                  state.v,
                                  state_update.v,
                                  gate.v,

                                  size);
  #endif
}


__host__
void cpu_gru_gate_backward_kernel(  float *error,
                                    float *state,
                                    float *state_update,
                                    float *gate,
                                    float *error_back_state,
                                    float *error_back_state_update,
                                    float *error_back_gate,
                                    unsigned int size)
{
  for (unsigned int idx = 0; idx < size; idx++)
  {
    float _error = error[idx];

    float _state         = state[idx];
    float _state_update  = SIGMOID(state_update[idx]);
    float _gate          = SIGMOID(gate[idx]);

    float _state_der          = 1.0;
    float _state_update_der   = SIGMOID_DERIVATIVE(state_update[idx]);
    float _gate_der           = SIGMOID_DERIVATIVE(gate[idx]);

    float _error_back_state         = _error*(1.0 - _gate)*_state_der;
    float _error_back_state_update  = _error*_gate*_state_update_der;
    float _error_back_gate          = _error*_state*(-_gate_der) + _error*_state_update*_gate_der;

    error_back_state[idx]         = _error_back_state;
    error_back_state_update[idx]  = _error_back_state_update;
    error_back_gate[idx]          = _error_back_gate;
  }
}

__global__
void cuda_gru_gate_backward_kernel( float *error,
                                    float *state,
                                    float *state_update,
                                    float *gate,
                                    float *error_back_state,
                                    float *error_back_state_update,
                                    float *error_back_gate,
                                    unsigned int size)
{
  unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

  if (idx < size)
  {
    float _error = error[idx];

    float _state         = state[idx];
    float _state_update  = SIGMOID(state_update[idx]);
    float _gate          = SIGMOID(gate[idx]);

    float _state_der          = 1.0;
    float _state_update_der   = SIGMOID_DERIVATIVE(state_update[idx]);
    float _gate_der           = SIGMOID_DERIVATIVE(gate[idx]);

    float _error_back_state         = _error*(1.0 - _gate)*_state_der;
    float _error_back_state_update  = _error*_gate*_state_update_der;
    float _error_back_gate          = _error*_state*(-_gate_der) + _error*_state_update*_gate_der;

    error_back_state[idx]         = _error_back_state;
    error_back_state_update[idx]  = _error_back_state_update;
    error_back_gate[idx]          = _error_back_gate;
  }
}

void gru_gate_backward( Tensor &error,
                        Tensor &state,
                        Tensor &state_update,
                        Tensor &gate,
                        Tensor &error_back_state,
                        Tensor &error_back_state_update,
                        Tensor &error_back_gate)
{
  unsigned int size = error.size();

  #ifdef NETWORK_USE_CUDA
      unsigned int block_size = 16;
      if (size >= 256)
        block_size = 64;

      dim3 block(block_size);
      dim3 grid((size + block.x - 1)/block.x);

      cuda_gru_gate_backward_kernel<<<grid, block>>>( error.v,

                                                      state.v,
                                                      state_update.v,
                                                      gate.v,

                                                      error_back_state.v,
                                                      error_back_state_update.v,
                                                      error_back_gate.v,

                                                      size);
      cudaDeviceSynchronize();
  #else
    cpu_gru_gate_backward_kernel( error.v,

                                  state.v,
                                  state_update.v,
                                  gate.v,

                                  error_back_state.v,
                                  error_back_state_update.v,
                                  error_back_gate.v,

                                  size);
  #endif
}
