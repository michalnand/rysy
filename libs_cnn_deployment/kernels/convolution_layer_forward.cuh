#ifndef _CONVOLUTION_LAYER_FORWARD_CUH_
#define _CONVOLUTION_LAYER_FORWARD_CUH_

#include <nn_struct.h>

void convolution_layer_forward( float *output, float *input,
                                float *weights, float *bias,
                                sGeometry input_geometry,
                                sGeometry kernel_geometry,
                                sGeometry output_geometry );


#endif
