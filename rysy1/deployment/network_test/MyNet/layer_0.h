#ifndef _LAYER_layer_0_H_
#define _LAYER_layer_0_H_


#include <NetworkConfig.h>


#define layer_0_type "convolution"

sLayerGeometry layer_0_input_geometry = {28, 28, 1};
sLayerGeometry layer_0_output_geometry = {28, 28, 8};
sLayerGeometry layer_0_kernel_geometry = {3, 3, 8};

#define layer_0_weights_size ((unsigned int)72) //array size
#define layer_0_weights_range ((nn_t)330) //multiply neuron result with range/1024

const nn_weight_t layer_0_weights[]={
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
-90, 95, -27, 55, -79, -32, 103, 90, 34, -41, -32, 20, 12, -96, 78, 14, 
-73, 54, 83, 5, 31, 40, 83, 50, -3, 19, 34, -66, -84, -87, 1, 60, 
-38, 36, 84, 36, 86, 31, -84, -127, -42, -41, 50, 15, 98, 74, 49, -37, 
-116, -8, 64, -2, -63, 72, 55, 19, };




#define layer_0_bias_size ((unsigned int)8) //array size
#define layer_0_bias_range ((nn_t)79) //multiply neuron result with range/1024

const nn_weight_t layer_0_bias[]={
-18, -15, 3, -127, -1, 2, 34, 35, };


#endif
