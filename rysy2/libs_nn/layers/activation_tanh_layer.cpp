#include <layers/activation_tanh_layer.h>

#include <kernels/activation_tanh_layer.cuh>

ActivationTanhLayer::ActivationTanhLayer()
        :Layer()
{

}

ActivationTanhLayer::ActivationTanhLayer(ActivationTanhLayer& other)
        :Layer(other)
{

}

ActivationTanhLayer::ActivationTanhLayer(const ActivationTanhLayer& other)
        :Layer(other)
{

}

ActivationTanhLayer::ActivationTanhLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_activation_tanh_layer();
}

ActivationTanhLayer::~ActivationTanhLayer()
{

}

ActivationTanhLayer& ActivationTanhLayer::operator= (ActivationTanhLayer& other)
{
    copy(other);
    return *this;
}

ActivationTanhLayer& ActivationTanhLayer::operator= (const ActivationTanhLayer& other)
{
    copy(other);
    return *this;
}


std::string ActivationTanhLayer::asString()
{
    std::string result;

    result+= "TANH\t\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}


void ActivationTanhLayer::forward(Tensor &output, Tensor &input)
{
    activation_tanh_layer_forward(output, input);
}

void ActivationTanhLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights)
{
    (void)input;
    (void)update_weights;

    activation_tanh_layer_backward(error_back, output, error);
}


void ActivationTanhLayer::init_activation_tanh_layer()
{
    this->m_output_shape            = this->m_input_shape;
    this->m_trainable_parameters    = 0;
    this->m_flops                   = 2*this->m_input_shape.size();
}
