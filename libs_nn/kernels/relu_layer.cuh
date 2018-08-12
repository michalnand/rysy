#ifndef _RELU_LAYER_CUH_
#define _RELU_LAYER_CUH_

#include "../tensor.h"

void relu_layer_forward(  Tensor &output, Tensor &input);

void relu_layer_backward( Tensor &error_back, Tensor &output, Tensor &error);

#endif
