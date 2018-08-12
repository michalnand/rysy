#ifndef _FC_LAYER_CUH_
#define _FC_LAYER_CUH_

#include "../tensor.h"

void fc_layer_forward(  Tensor &output, Tensor &input,
                        Tensor &w, Tensor &bias);

void fc_layer_backward( Tensor &error_back, Tensor &input, Tensor &error, Tensor &w);

void fc_layer_gradient(Tensor &w_grad, Tensor &input, Tensor &error);
void fc_layer_update_bias(Tensor &bias, Tensor &error, float learning_rate);

#endif
