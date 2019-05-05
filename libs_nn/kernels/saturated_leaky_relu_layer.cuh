#ifndef _SATURATED_LEAKY_RELU_LAYER_CUH_
#define _SATURATED_LEAKY_RELU_LAYER_CUH_

#include "../tensor.h"

void saturated_leaky_relu_layer_forward(  Tensor &output, Tensor &input);

void saturated_leaky_relu_layer_backward( Tensor &error_back, Tensor &output, Tensor &error);

#endif
