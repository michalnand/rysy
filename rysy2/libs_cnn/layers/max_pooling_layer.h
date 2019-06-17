#ifndef _MAX_POOLING_LAYER_H_
#define _MAX_POOLING_LAYER_H_

#include <layers/layer.h>

class MaxPoolingLayer final: public Layer
{
    public:
        MaxPoolingLayer();
        MaxPoolingLayer(MaxPoolingLayer& other);
        MaxPoolingLayer(const MaxPoolingLayer& other);

        MaxPoolingLayer(Shape input_shape, Json::Value parameters);

        virtual ~MaxPoolingLayer();

        MaxPoolingLayer& operator= (MaxPoolingLayer& other);
        MaxPoolingLayer& operator= (const MaxPoolingLayer& other);

        std::string asString();

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights);


    protected:
        void init_max_pooling_layer();

    private:
        Tensor max_mask;
};

#endif
