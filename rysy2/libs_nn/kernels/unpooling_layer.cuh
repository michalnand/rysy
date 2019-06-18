#ifndef _UNPOOLING_LAYER_CUH_
#define _UNPOOLING_LAYER_CUH_

#include <tensor.h>

void unpooling_layer_forward(Tensor &output, Tensor &input);

void unpooling_layer_backward(Tensor &error_back, Tensor &error);

#endif
