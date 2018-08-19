#ifndef _LAYER_layer_0_H_
#define _LAYER_layer_0_H_


#include <NetworkConfig.h>


#define layer_0_type "convolution"

sLayerGeometry layer_0_input_geometry = {28, 28, 1};
sLayerGeometry layer_0_output_geometry = {28, 28, 8};
sLayerGeometry layer_0_kernel_geometry = {3, 3, 8};

#define layer_0_weights_size ((unsigned int)72) //array size
#define layer_0_weights_range ((nn_t)383) //multiply neuron result with range/1024

const nn_weight_t layer_0_weights[]={
0, -22, 0, 0, -23, 0, 0, -24, 0, -70, -15, -79, 0, 43, 37, 0, 
82, 30, 11, 51, -14, 54, 54, -78, 86, -38, -112, 39, -12, -4, 78, -53, 
-88, 48, 60, 73, -127, -61, 86, 15, 73, 43, 56, 11, -30, 86, 82, -65, 
-12, 92, 50, -8, 41, 29, 76, 84, 91, -7, -35, -26, -73, -67, -44, -46, 
55, 25, -35, 28, 107, -52, -88, 4, };




#define layer_0_bias_size ((unsigned int)8) //array size
#define layer_0_bias_range ((nn_t)9) //multiply neuron result with range/1024

const nn_weight_t layer_0_bias[]={
0, 0, -127, 11, -124, -10, -10, -1, };


#endif
