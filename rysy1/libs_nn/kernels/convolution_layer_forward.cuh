#ifndef _CONVOLUTION_LAYER_FORWARD_CUH_
#define _CONVOLUTION_LAYER_FORWARD_CUH_

#include "../tensor.h"

void convolution_layer_forward(   Tensor &output, Tensor &input,
                                  Tensor &w, Tensor &bias);


#endif
