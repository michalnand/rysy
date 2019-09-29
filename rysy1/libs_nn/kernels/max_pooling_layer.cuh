#ifndef _MAX_POOLING_LAYER_CUH_
#define _MAX_POOLING_LAYER_CUH_

#include "../tensor.h"

void max_pooling_layer_forward(Tensor &max_mask, Tensor &output, Tensor &input);

void max_pooling_layer_backward(Tensor &error_back, Tensor &error, Tensor &max_mask);

#endif
