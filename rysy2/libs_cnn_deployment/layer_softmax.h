#ifndef _LAYER_SOFTMAX_H_
#define _LAYER_SOFTMAX_H_

#include <layer_interface.h>

class LayerSoftmax: public LayerInterface
{
    public:
        LayerSoftmax();
        LayerSoftmax(LayerSoftmax& other);
        LayerSoftmax(const LayerSoftmax& other);

        LayerSoftmax(Json::Value json, sShape input_shape = {0, 0, 0});

        virtual ~LayerSoftmax();

        LayerSoftmax& operator= (LayerSoftmax& other);
        LayerSoftmax& operator= (const LayerSoftmax& other);

    protected:
        void copy(LayerSoftmax& other);
        void copy(const LayerSoftmax& other);

    public:
        void print();
        void forward(float *output, float *input);
};

#endif
