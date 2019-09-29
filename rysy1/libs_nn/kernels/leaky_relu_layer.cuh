#ifndef _LEAKY_RELU_LAYER_CUH_
#define _LEAKY_RELU_LAYER_CUH_

#include "../tensor.h"

void leaky_relu_layer_forward(  Tensor &output, Tensor &input);

void leaky_relu_layer_backward( Tensor &error_back, Tensor &output, Tensor &error);

#endif
