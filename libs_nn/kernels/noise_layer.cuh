#ifndef _NOISE_LAYER_CUH_
#define _NOISE_LAYER_CUH_

#include "../tensor.h"

void noise_layer_forward(
                          Tensor &output, Tensor &input, sHyperparameters hyperparameters,
                          Tensor &white_noise, Tensor &salt_and_pepper_noise, float brightness_noise
                        );


#endif
