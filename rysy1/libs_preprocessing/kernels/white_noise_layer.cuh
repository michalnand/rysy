#ifndef _WHITE_NOISE_LAYER_CUH_
#define _WHITE_NOISE_LAYER_CUH_

#include <tensor.h>

void white_noise_layer(Tensor &output, Tensor &input, Tensor &noise, float noise_level);


#endif
