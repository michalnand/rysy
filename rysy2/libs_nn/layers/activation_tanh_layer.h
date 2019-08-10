#ifndef _ACTIVATION_TANH_LAYER_H_
#define _ACTIVATION_TANH_LAYER_H_

#include <layers/layer.h>

class ActivationTanhLayer final: public Layer
{
    public:
        ActivationTanhLayer();
        ActivationTanhLayer(ActivationTanhLayer& other);
        ActivationTanhLayer(const ActivationTanhLayer& other);

        ActivationTanhLayer(Shape input_shape, Json::Value parameters);

        virtual ~ActivationTanhLayer();

        ActivationTanhLayer& operator= (ActivationTanhLayer& other);
        ActivationTanhLayer& operator= (const ActivationTanhLayer& other);

        std::string asString();

        bool is_activation() {return true;}

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights);


    protected:
        void init_activation_tanh_layer();
};

#endif
