#ifndef _EMBEDDED_NET_RELU_LAYER_H_
#define _EMBEDDED_NET_RELU_LAYER_H_

#include <EmbeddedNetLayer.h>

class EmbeddedNetReluLayer: public EmbeddedNetLayer
{
    private:
        unsigned int size;

    public:
        EmbeddedNetReluLayer(sEmbeddedNetShape kernel_shape, sEmbeddedNetShape input_shape, sEmbeddedNetShape output_shape);
        virtual ~EmbeddedNetReluLayer();

        void forward(nn_layer_t *output, nn_layer_t *input);
};

#endif
