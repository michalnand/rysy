#ifndef _UNPOOLING_LAYER_H_
#define _UNPOOLING_LAYER_H_

#include <layers/layer.h>

class UnPoolingLayer final: public Layer
{
    public:
        UnPoolingLayer();
        UnPoolingLayer(UnPoolingLayer& other);
        UnPoolingLayer(const UnPoolingLayer& other);

        UnPoolingLayer(Shape input_shape, Json::Value parameters);

        virtual ~UnPoolingLayer();

        UnPoolingLayer& operator= (UnPoolingLayer& other);
        UnPoolingLayer& operator= (const UnPoolingLayer& other);

        std::string asString();

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias = true);


    protected:
        void init_average_pooling_layer();

};

#endif
