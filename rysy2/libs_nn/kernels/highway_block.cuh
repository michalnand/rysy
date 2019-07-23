#ifndef _HIGHWAY_LAYER_CUH_
#define _HIGHWAY_LAYER_CUH_

#include <tensor.h>

void highway_layer_forward(Tensor &output, Tensor &input);
void highway_layer_backward(Tensor &error_back, Tensor &input, Tensor &error);

#endif
