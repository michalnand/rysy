#ifndef _TANH_LAYER_CUH_
#define _TANH_LAYER_CUH_

#include "../tensor.h"

void tanh_layer_forward(  Tensor &output, Tensor &input);

void tanh_layer_backward( Tensor &error_back, Tensor &output, Tensor &error);

#endif
