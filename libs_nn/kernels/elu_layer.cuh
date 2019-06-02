#ifndef _ELU_LAYER_CUH_
#define _ELU_LAYER_CUH_

#include "../tensor.h"

void elu_layer_forward(  Tensor &output, Tensor &input);

void elu_layer_backward( Tensor &error_back, Tensor &output, Tensor &error);

#endif
