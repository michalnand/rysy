#ifndef _CROP_LAYER_CUH_
#define _CROP_LAYER_CUH_

#include <tensor.h>

void crop_layer_forward(Tensor &output, Tensor &input);

void crop_layer_backward(Tensor &error_back, Tensor &error);

#endif
