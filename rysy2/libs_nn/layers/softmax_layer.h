#ifndef _SOFTMAX_LAYER_H_
#define _SOFTMAX_LAYER_H_

#include <layers/layer.h>

class SoftmaxLayer final: public Layer
{
    public:
        SoftmaxLayer();
        SoftmaxLayer(SoftmaxLayer& other);
        SoftmaxLayer(const SoftmaxLayer& other);

        SoftmaxLayer(Shape input_shape, Json::Value parameters);

        virtual ~SoftmaxLayer();

        SoftmaxLayer& operator= (SoftmaxLayer& other);
        SoftmaxLayer& operator= (const SoftmaxLayer& other);

        std::string asString();

        bool is_activation() {return true;}

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias = true);


    protected:
        void init_activation_elu_layer();
};

#endif
