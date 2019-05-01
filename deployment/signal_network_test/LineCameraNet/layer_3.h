#ifndef _LAYER_layer_3_H_
#define _LAYER_layer_3_H_


#include <NetworkConfig.h>


#define layer_3_type "convolution"

const sLayerGeometry layer_3_input_geometry = {64, 1, 4};
const sLayerGeometry layer_3_output_geometry = {64, 1, 4};
const sLayerGeometry layer_3_kernel_geometry = {3, 1, 4};

#define layer_3_weights_size ((unsigned int)48) //array size
#define layer_3_weights_range ((nn_t)1386) //multiply neuron result with range/1024

const nn_weight_t layer_3_weights[]={
0, -19, 17, 25, 34, 29, 25, 6, 28, -84, -98, -84, -29, 5, -16, 32, 
5, 11, 21, 33, 21, -65, -87, -74, -18, -15, -15, -15, 0, -8, 3, -11, 
-2, 14, -12, 0, 6, 3, 7, 2, 4, 0, 0, -4, -2, 127, 52, 121, 
};




#define layer_3_bias_size ((unsigned int)4) //array size
#define layer_3_bias_range ((nn_t)185) //multiply neuron result with range/1024

const nn_weight_t layer_3_bias[]={
-127, -70, 0, 39, };


#endif
