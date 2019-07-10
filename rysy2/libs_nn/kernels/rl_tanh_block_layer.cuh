#ifndef _RL_TANH_BLOCK_LAYER_CUH_
#define _RL_TANH_BLOCK_LAYER_CUH_

#include <tensor.h>

void rl_tanh_block_layer_forward(  Tensor &output, Tensor &inputx, Tensor &inputh,
                             Tensor &wx, Tensor &wh, Tensor &bias);

void rl_tanh_block_layer_backward(  Tensor &error_back_x, Tensor &error_back_h,
                                    Tensor &inputx, Tensor &inputh,
                                    Tensor &output, Tensor &error,
                                    Tensor &wx, Tensor &wh);

void rl_tanh_block_layer_gradient(  Tensor &wx_grad, Tensor &wh_grad,
                                    Tensor &inputx, Tensor &inputh,
                                    Tensor &output, Tensor &error, Tensor &error_h);


void rl_tanh_block_layer_update_bias(Tensor &bias, Tensor &output, Tensor &error, float learning_rate);

#endif
