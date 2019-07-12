#ifndef _SOFTMAX_LAYER_FORWARD_CUH_
#define _SOFTMAX_LAYER_FORWARD_CUH_

#include <nn_struct.h>

void softmax_layer_forward( float *output, float *input,
                            sShape input_shape);


#endif
