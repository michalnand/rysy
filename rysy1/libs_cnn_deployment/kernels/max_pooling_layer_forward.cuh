#ifndef _MAX_POOLING_LAYER_FORWARD_CUH_
#define _MAX_POOLING_LAYER_FORWARD_CUH_

#include <nn_struct.h>

void max_pooling_layer_forward( float *output,
                                float *input,
                                sGeometry input_geometry);

#endif
