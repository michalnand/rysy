#ifndef _DROPOUT_LAYER_CUH_
#define _DROPOUT_LAYER_CUH_

#include "../tensor.h"

void dropout_layer_forward(  Tensor &output, Tensor &input, Tensor &noise, float dropout);

void dropout_layer_backward( Tensor &error_back, Tensor &output, Tensor &error);

#endif
