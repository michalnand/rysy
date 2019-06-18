#ifndef _LAYER_MAX_POOLING_H_
#define _LAYER_MAX_POOLING_H_

#include <layer_interface.h>

class LayerMaxPooling: public LayerInterface
{
    public:
        LayerMaxPooling();
        LayerMaxPooling(LayerMaxPooling& other);
        LayerMaxPooling(const LayerMaxPooling& other);

        LayerMaxPooling(Json::Value json, sShape input_shape = {0, 0, 0});

        virtual ~LayerMaxPooling();

        LayerMaxPooling& operator= (LayerMaxPooling& other);
        LayerMaxPooling& operator= (const LayerMaxPooling& other);

    protected:
        void copy(LayerMaxPooling& other);
        void copy(const LayerMaxPooling& other);

    public:
        void print();
        void forward(float *output, float *input);


};

#endif
