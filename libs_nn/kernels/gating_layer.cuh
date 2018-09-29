#ifndef _GATING_LAYER_CUH_
#define _GATING_LAYER_CUH_

#include "../tensor.h"

void gating_layer_forward(  Tensor &output, Tensor &input);

void gating_layer_backward( Tensor &error_back, Tensor &input, Tensor &output, Tensor &error);

#endif
