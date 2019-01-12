#ifndef _RELU_6_LAYER_CUH_
#define _RELU_6_LAYER_CUH_

#include "../tensor.h"

void relu_6_layer_forward(  Tensor &output, Tensor &input);

void relu_6_layer_backward( Tensor &error_back, Tensor &output, Tensor &error);

#endif
