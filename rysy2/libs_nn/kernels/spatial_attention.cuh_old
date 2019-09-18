#ifndef _SPATIAL_ATTENTION_CUH_
#define _SPATIAL_ATTENTION_CUH_

#include <tensor.h>

void spatial_attention_forward(  Tensor &output,
                                 Tensor &input,
                                 Tensor &input_attention);

void spatial_attention_backward(  Tensor &error_back,
                                  Tensor &error_back_attention,
                                  Tensor &input,
                                  Tensor &input_attention,
                                  Tensor &error);


#endif
