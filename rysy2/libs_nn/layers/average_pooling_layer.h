#ifndef _AVERAGE_POOLING_LAYER_H_
#define _AVERAGE_POOLING_LAYER_H_

#include <layers/layer.h>

class AveragePoolingLayer final: public Layer
{
    public:
        AveragePoolingLayer();
        AveragePoolingLayer(AveragePoolingLayer& other);
        AveragePoolingLayer(const AveragePoolingLayer& other);

        AveragePoolingLayer(Shape input_shape, Json::Value parameters);

        virtual ~AveragePoolingLayer();

        AveragePoolingLayer& operator= (AveragePoolingLayer& other);
        AveragePoolingLayer& operator= (const AveragePoolingLayer& other);

        std::string asString();

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias = true);


    protected:
        void init_average_pooling_layer();

};

#endif
