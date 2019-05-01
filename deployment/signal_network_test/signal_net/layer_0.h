#ifndef _LAYER_layer_0_H_
#define _LAYER_layer_0_H_


#include <NetworkConfig.h>


#define layer_0_type "convolution"

const sLayerGeometry layer_0_input_geometry = {128, 1, 1};
const sLayerGeometry layer_0_output_geometry = {128, 1, 8};
const sLayerGeometry layer_0_kernel_geometry = {3, 1, 8};

#define layer_0_weights_size ((unsigned int)24) //array size
#define layer_0_weights_range ((nn_t)1380) //multiply neuron result with range/1024

const nn_weight_t layer_0_weights[]={
-3, 0, 0, 72, 92, 3, -17, -2, 0, 5, 118, 50, 67, 90, 32, -30, 
-127, -30, 56, 13, 108, 26, 108, 53, };




#define layer_0_bias_size ((unsigned int)8) //array size
#define layer_0_bias_range ((nn_t)1228) //multiply neuron result with range/1024

const nn_weight_t layer_0_bias[]={
0, -28, 0, -32, -28, 127, -30, -32, };


#endif
