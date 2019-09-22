#ifndef _SOFTMAX_CUH_
#define _SOFTMAX_CUH_

#include <tensor.h>

void softmax_layer_forward(  Tensor &output,
                            Tensor &input);

void softmax_layer_backward( Tensor &error_back,
                            Tensor &output,
                            Tensor &error);


#endif
