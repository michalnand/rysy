#ifndef _ATTENTION_GATE_LAYER_CUH_
#define _ATTENTION_GATE_LAYER_CUH_

#include <tensor.h>

void attention_gate_layer_forward(  Tensor &output, Tensor &input);

void attention_gate_layer_backward( Tensor &error_back, Tensor &output, Tensor &error);

#endif
