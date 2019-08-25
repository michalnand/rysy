#ifndef _FLATTEN_LAYER_H_
#define _FLATTEN_LAYER_H_

#include <layers/layer.h>

class FlattenLayer final: public Layer
{
    public:
        FlattenLayer();
        FlattenLayer(FlattenLayer& other);
        FlattenLayer(const FlattenLayer& other);

        FlattenLayer(Shape input_shape, Json::Value parameters);

        virtual ~FlattenLayer();

        FlattenLayer& operator= (FlattenLayer& other);
        FlattenLayer& operator= (const FlattenLayer& other);

        std::string asString();

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias = true);


    protected:
        void init_flatten_layer();
};

#endif
