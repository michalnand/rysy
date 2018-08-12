#ifndef _CONVOLUTION_LAYER_CUH_
#define _CONVOLUTION_LAYER_CUH_

#include "../tensor.h"

void convolution_layer_forward(   Tensor &output, Tensor &input,
                                  Tensor &w, Tensor &bias);

void convolution_layer_gradient(Tensor &w_grad, Tensor &input, Tensor &error);

void convolution_layer_update_bias(Tensor &bias, Tensor &error, float learning_rate);


void convolution_layer_backward( Tensor &error_back, Tensor &input, Tensor &error, Tensor &w);



#endif
