#ifndef _LAYER_layer_3_H_
#define _LAYER_layer_3_H_


#include <NetworkConfig.h>


#define layer_3_type "convolution"

const sLayerGeometry layer_3_input_geometry = {64, 1, 8};
const sLayerGeometry layer_3_output_geometry = {64, 1, 8};
const sLayerGeometry layer_3_kernel_geometry = {3, 1, 8};

#define layer_3_weights_size ((unsigned int)192) //array size
#define layer_3_weights_range ((nn_t)1945) //multiply neuron result with range/1024

const nn_weight_t layer_3_weights[]={
4, -4, -1, 3, 1, -1, 5, 0, -1, 0, -4, -7, 0, 0, -7, 0, 
-2, -4, 0, -6, -5, 0, -7, -5, -1, -3, 0, 11, 32, 31, 0, 0, 
5, 23, 34, 41, 21, 28, 26, -86, -92, -107, 15, 28, 37, 15, 32, 38, 
1, -6, 5, -1, 2, 0, -2, -3, -4, 1, -4, 2, -6, 5, 2, 0, 
-2, -6, -2, -1, -3, -3, 3, -6, -7, -4, -6, -2, -4, 2, 1, -4, 
2, -3, 0, 0, -6, 1, 1, -1, -7, 0, -6, 0, 1, 4, -8, 0, 
-3, 2, 0, 35, 32, 2, 0, 2, 2, 37, 24, -10, 40, 16, 1, -127, 
-119, -57, 35, 22, -11, 36, 22, 0, 2, 0, -4, 2, 3, -2, 4, 5, 
4, 8, -6, -9, 5, -5, -1, 6, 13, 16, 4, -10, -10, 0, -7, -6, 
-2, 0, 2, -15, 13, 26, -4, -1, 3, -6, 15, 25, -11, 9, 28, 0, 
-38, -83, -5, 22, 27, -8, 20, 29, 4, -5, 1, 39, 27, 18, 1, 5, 
0, 33, 35, 21, 41, 27, 16, -84, -78, -64, 40, 19, 22, 30, 22, 24, 
};




#define layer_3_bias_size ((unsigned int)8) //array size
#define layer_3_bias_range ((nn_t)122) //multiply neuron result with range/1024

const nn_weight_t layer_3_bias[]={
0, -124, 0, 0, -39, 22, -127, -39, };


#endif
