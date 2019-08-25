#ifndef _ACTIVATION_RELU_LAYER_H_
#define _ACTIVATION_RELU_LAYER_H_

#include <layers/layer.h>

class ActivationReluLayer final: public Layer
{
    public:
        ActivationReluLayer();
        ActivationReluLayer(ActivationReluLayer& other);
        ActivationReluLayer(const ActivationReluLayer& other);

        ActivationReluLayer(Shape input_shape, Json::Value parameters);

        virtual ~ActivationReluLayer();

        ActivationReluLayer& operator= (ActivationReluLayer& other);
        ActivationReluLayer& operator= (const ActivationReluLayer& other);

        std::string asString();

        bool is_activation() {return true;}

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias = true);


    protected:
        void init_activation_relu_layer();
};

#endif
