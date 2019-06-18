#ifndef _LAYER_CONVOLUTION_H_
#define _LAYER_CONVOLUTION_H_

#include <layer_interface.h>

class LayerConvolution: public LayerInterface
{
    public:
        LayerConvolution();
        LayerConvolution(LayerConvolution& other);
        LayerConvolution(const LayerConvolution& other);

        LayerConvolution(Json::Value json, sShape input_shape = {0, 0, 0});

        virtual ~LayerConvolution();

        LayerConvolution& operator= (LayerConvolution& other);
        LayerConvolution& operator= (const LayerConvolution& other);

    protected:
        void copy(LayerConvolution& other);
        void copy(const LayerConvolution& other);

    public:
        void print();
        void forward(float *output, float *input);

    private:
        float *weights, *bias;

};

#endif
