#ifndef _ACTIVATION_ELU_LAYER_CUH_
#define _ACTIVATION_ELU_LAYER_CUH_

#include <tensor.h>

void activation_elu_layer_forward(  Tensor &output, Tensor &input);

void activation_elu_layer_backward( Tensor &error_back, Tensor &output, Tensor &error);

#endif
