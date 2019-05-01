#ifndef _LAYER_layer_0_H_
#define _LAYER_layer_0_H_


#include <NetworkConfig.h>


#define layer_0_type "convolution"

const sLayerGeometry layer_0_input_geometry = {128, 1, 1};
const sLayerGeometry layer_0_output_geometry = {128, 1, 4};
const sLayerGeometry layer_0_kernel_geometry = {3, 1, 4};

#define layer_0_weights_size ((unsigned int)12) //array size
#define layer_0_weights_range ((nn_t)1375) //multiply neuron result with range/1024

const nn_weight_t layer_0_weights[]={
-18, -2, 19, -19, 58, 40, 53, 21, 8, 1, -127, 1, };




#define layer_0_bias_size ((unsigned int)4) //array size
#define layer_0_bias_range ((nn_t)324) //multiply neuron result with range/1024

const nn_weight_t layer_0_bias[]={
115, -67, -54, 127, };


#endif
