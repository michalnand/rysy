#ifndef _SPATIAL_ATTENTION_CUH_
#define _SPATIAL_ATTENTION_CUH_

#include <tensor.h>

void spatial_attention_forward(  Tensor &output,
                                 Tensor &input);

void spatial_attention_backward(  Tensor &error_back,
                                  Tensor &input,
                                  Tensor &error);


#endif
