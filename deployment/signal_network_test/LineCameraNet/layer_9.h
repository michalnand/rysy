#ifndef _LAYER_layer_9_H_
#define _LAYER_layer_9_H_


#include <NetworkConfig.h>


#define layer_9_type "convolution"

const sLayerGeometry layer_9_input_geometry = {16, 1, 8};
const sLayerGeometry layer_9_output_geometry = {16, 1, 8};
const sLayerGeometry layer_9_kernel_geometry = {3, 1, 8};

#define layer_9_weights_size ((unsigned int)192) //array size
#define layer_9_weights_range ((nn_t)1124) //multiply neuron result with range/1024

const nn_weight_t layer_9_weights[]={
27, 12, -28, 49, 22, 5, 16, -4, -59, -10, 3, -6, 0, -15, 2, -9, 
-10, -1, 8, -11, -12, 16, 12, -35, -5, 17, 4, -22, -4, 4, -17, 24, 
9, 0, -14, -15, 1, -17, -1, 2, 6, 11, -34, -22, 0, 12, 34, 3, 
30, -4, -23, 63, 11, 2, 6, 0, -29, -127, -9, 7, -101, -7, -6, 0, 
-5, -2, -52, 10, 4, 26, -13, -42, 13, 32, 9, -23, -34, -15, 6, 39, 
18, 12, -12, -4, 4, -19, 1, 10, -3, 1, -33, -15, -1, 12, 25, -7, 
-10, -1, -1, -3, -13, -3, -1, 1, -3, 6, 4, 2, -12, -10, 5, 1, 
11, -4, -11, -8, -3, -4, -3, 2, 0, 35, 12, -4, -15, -2, -6, 26, 
6, 4, -19, -6, 7, -11, -11, 10, -6, -1, -33, -13, 1, 7, 30, -2, 
-49, 10, 21, 0, -19, 15, -82, 12, 14, 6, -19, -69, 1, -15, -71, -4, 
0, 8, 7, -83, -12, -67, 27, 23, 26, 8, -31, 10, 20, 12, 7, 21, 
-64, -16, -68, -1, -21, -64, 3, 2, -9, 5, -96, -27, 9, 15, 21, -59, 
};




#define layer_9_bias_size ((unsigned int)8) //array size
#define layer_9_bias_range ((nn_t)266) //multiply neuron result with range/1024

const nn_weight_t layer_9_bias[]={
52, 127, 79, 87, 0, 100, 118, 100, };


#endif
