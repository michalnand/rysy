#ifndef _LAYER_layer_6_H_
#define _LAYER_layer_6_H_


#include <NetworkConfig.h>


#define layer_6_type "convolution"

const sLayerGeometry layer_6_input_geometry = {16, 1, 8};
const sLayerGeometry layer_6_output_geometry = {16, 1, 8};
const sLayerGeometry layer_6_kernel_geometry = {3, 1, 8};

#define layer_6_weights_size ((unsigned int)192) //array size
#define layer_6_weights_range ((nn_t)840) //multiply neuron result with range/1024

const nn_weight_t layer_6_weights[]={
-17, -69, -22, 24, -25, 39, 34, 27, 34, 15, 44, 5, 11, 29, 9, 10, 
50, -5, 2, -16, -12, -2, -61, -41, -9, -18, -7, 10, -6, -5, -12, -14, 
-4, -12, -8, 1, -6, -16, -3, -3, 3, -11, -3, -6, 10, 8, 9, -17, 
22, -5, 0, -76, -46, 20, 0, 18, 24, -46, -21, 39, -24, 0, 13, -77, 
21, 44, 0, 5, 10, 46, 42, -21, 21, 7, 5, -42, -38, 37, 0, 17, 
9, -58, -7, 43, -38, 18, 7, -60, 17, 26, -16, -17, 2, 11, 28, -38, 
-38, -18, 13, -13, -60, -27, -16, -2, 5, 67, 29, -98, 72, 21, -65, 52, 
3, -84, -17, -3, -5, -89, -5, 34, -23, -68, -20, 0, -28, 33, 27, 35, 
20, -12, 22, 15, -16, 49, 5, 35, 44, 34, 13, -9, 6, 13, -60, -32, 
-8, -8, -13, -3, 4, -10, 0, -12, -12, -8, 6, 9, -2, 7, -11, 5, 
-12, -14, 9, 11, -14, 0, -12, 0, 30, -80, -59, -48, -102, -7, 10, 5, 
18, -82, -15, 49, -60, 19, 30, -127, 30, 82, 2, -6, -9, 28, -12, -90, 
};




#define layer_6_bias_size ((unsigned int)8) //array size
#define layer_6_bias_range ((nn_t)172) //multiply neuron result with range/1024

const nn_weight_t layer_6_bias[]={
119, 0, 38, 71, 127, 108, 0, 69, };


#endif
