#ifndef _LAYER_layer_0_H_
#define _LAYER_layer_0_H_


#include <NetworkConfig.h>


#define layer_0_type "convolution"

sLayerGeometry layer_0_input_geometry = {28, 28, 1};
sLayerGeometry layer_0_output_geometry = {28, 28, 8};
sLayerGeometry layer_0_kernel_geometry = {3, 3, 8};

#define layer_0_weights_size ((unsigned int)72) //array size
#define layer_0_weights_range ((nn_t)276) //multiply neuron result with range/1024

const nn_weight_t layer_0_weights[]={
-92, -27, 90, -11, 0, 39, 30, -69, -124, 2, 30, 36, 3, -25, 51, -19, 
22, -14, -7, 85, 55, -56, -33, 21, -87, -44, 28, -5, 66, 13, 11, 0, 
32, -83, -16, -68, -127, -16, 12, -64, -119, 14, 99, -26, -48, -33, 22, 42, 
-80, -47, 65, -83, 4, 46, 32, -50, 9, 76, 27, -48, 64, -43, -65, 28, 
33, 52, 31, -18, -87, -12, -47, 41, };




#define layer_0_bias_size ((unsigned int)8) //array size
#define layer_0_bias_range ((nn_t)97) //multiply neuron result with range/1024

const nn_weight_t layer_0_bias[]={
-20, -16, -103, -127, -16, -21, -40, -94, };


#endif
