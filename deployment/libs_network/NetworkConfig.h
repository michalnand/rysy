#ifndef _NETWORK_CONFIG_H_
#define _NETWORK_CONFIG_H_

#include <stdint.h>

//#define ARM_CORTEX_M4_M7
//#define MULTI_THREADS_SUPPORT

struct sLayerGeometry
{
    unsigned int w, h, d;
};

typedef int8_t    nn_weight_t;
typedef int8_t    nn_layer_t;
typedef int32_t   nn_t;


#define NETWORK_LAYERS_MAX_COUNT    ((unsigned int)32)
#define NETWORK_LAYER_OUTPUT_RANGE  ((1 << (7*sizeof(nn_layer_t))) - 1)


#endif
