#ifndef _LAYER_layer_6_H_
#define _LAYER_layer_6_H_


#include <NetworkConfig.h>


#define layer_6_type "convolution"

const sLayerGeometry layer_6_input_geometry = {32, 1, 4};
const sLayerGeometry layer_6_output_geometry = {32, 1, 8};
const sLayerGeometry layer_6_kernel_geometry = {3, 1, 8};

#define layer_6_weights_size ((unsigned int)96) //array size
#define layer_6_weights_range ((nn_t)685) //multiply neuron result with range/1024

const nn_weight_t layer_6_weights[]={
59, 47, 12, 54, 66, 34, 26, 13, 19, -67, -81, -61, 57, 35, -35, 64, 
40, -11, 10, 27, -2, -57, 15, 46, 70, 35, 32, 40, 64, 65, 10, -22, 
14, -72, -127, -104, -47, -8, -2, -56, -30, -10, 17, 0, -25, 120, 68, 68, 
-48, -23, -15, -57, -10, -13, 26, -22, 1, 99, 74, 93, 1, -26, -7, -6, 
10, 5, -12, -26, 10, 0, -21, 5, 29, -45, -28, 15, -58, -12, 26, 20, 
6, 60, 62, 65, 12, 50, 58, 59, 58, 35, -20, 13, -11, -102, -111, -72, 
};




#define layer_6_bias_size ((unsigned int)8) //array size
#define layer_6_bias_range ((nn_t)146) //multiply neuron result with range/1024

const nn_weight_t layer_6_bias[]={
-31, -53, -127, 84, 62, 0, -23, -52, };


#endif
