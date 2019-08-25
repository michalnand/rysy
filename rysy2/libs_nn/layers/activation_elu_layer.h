#ifndef _ACTIVATION_ELU_LAYER_H_
#define _ACTIVATION_ELU_LAYER_H_

#include <layers/layer.h>

class ActivationEluLayer final: public Layer
{
    public:
        ActivationEluLayer();
        ActivationEluLayer(ActivationEluLayer& other);
        ActivationEluLayer(const ActivationEluLayer& other);

        ActivationEluLayer(Shape input_shape, Json::Value parameters);

        virtual ~ActivationEluLayer();

        ActivationEluLayer& operator= (ActivationEluLayer& other);
        ActivationEluLayer& operator= (const ActivationEluLayer& other);

        std::string asString();

        bool is_activation() {return true;}

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias = true);


    protected:
        void init_activation_elu_layer();
};

#endif
