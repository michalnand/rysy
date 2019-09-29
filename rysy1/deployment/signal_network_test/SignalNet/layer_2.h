#ifndef _LAYER_layer_2_H_
#define _LAYER_layer_2_H_


#include <NetworkConfig.h>


#define layer_2_type "convolution"

const sLayerGeometry layer_2_input_geometry = {64, 1, 4};
const sLayerGeometry layer_2_output_geometry = {64, 1, 4};
const sLayerGeometry layer_2_kernel_geometry = {3, 1, 4};

#define layer_2_weights_size ((unsigned int)48) //array size
#define layer_2_weights_range ((nn_t)1314) //multiply neuron result with range/1024

const nn_weight_t layer_2_weights[]={
0, -12, -3, 73, 127, 76, 5, 13, 0, -3, -4, -7, -2, -9, 0, 92, 
104, 71, 36, 24, 16, -15, -17, -6, -10, 15, 8, -66, -87, -77, -29, -2, 
-12, 29, 46, 37, -10, 0, 3, -64, -53, -38, 19, 20, 3, 6, 34, 28, 
};




#define layer_2_bias_size ((unsigned int)4) //array size
#define layer_2_bias_range ((nn_t)205) //multiply neuron result with range/1024

const nn_weight_t layer_2_bias[]={
13, 15, -127, -10, };


#endif
