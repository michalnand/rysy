#ifndef _LAYER_layer_4_H_
#define _LAYER_layer_4_H_


#include <NetworkConfig.h>


#define layer_4_type "convolution"

const sLayerGeometry layer_4_input_geometry = {32, 1, 4};
const sLayerGeometry layer_4_output_geometry = {32, 1, 8};
const sLayerGeometry layer_4_kernel_geometry = {3, 1, 8};

#define layer_4_weights_size ((unsigned int)96) //array size
#define layer_4_weights_range ((nn_t)827) //multiply neuron result with range/1024

const nn_weight_t layer_4_weights[]={
37, 69, 78, 21, 35, 41, -18, -45, -76, 1, 7, -22, 53, 23, 73, 68, 
29, 58, -54, -33, -52, -27, -12, -29, -15, -1, 2, 44, 26, 18, -41, -10, 
-4, -12, 31, 16, -69, -112, -103, -77, -78, -71, 57, 62, 12, 56, 57, 20, 
-81, -97, -83, -21, -72, -45, 55, 69, 35, 47, 57, 26, -101, -127, -125, -27, 
-84, -67, -1, 59, 53, 17, 40, 63, -24, 12, 0, 9, -15, -11, 14, -3, 
-17, -25, 4, -13, 80, 29, 67, 70, 15, 74, -67, -19, -28, -29, -6, 10, 
};




#define layer_4_bias_size ((unsigned int)8) //array size
#define layer_4_bias_range ((nn_t)116) //multiply neuron result with range/1024

const nn_weight_t layer_4_bias[]={
-38, 9, 127, -76, 7, 33, -2, 115, };


#endif
