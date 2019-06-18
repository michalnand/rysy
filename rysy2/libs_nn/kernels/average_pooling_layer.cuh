#ifndef _AVERAGE_POOLING_LAYER_CUH_
#define _AVERAGE_POOLING_LAYER_CUH_

#include <tensor.h>

void average_pooling_layer_forward(Tensor &output, Tensor &input);

void average_pooling_layer_backward(Tensor &error_back, Tensor &error);

#endif
