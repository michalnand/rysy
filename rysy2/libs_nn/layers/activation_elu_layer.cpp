#include <layers/activation_elu_layer.h>

#include <kernels/activation_elu_layer.cuh>

ActivationEluLayer::ActivationEluLayer()
        :Layer()
{

}

ActivationEluLayer::ActivationEluLayer(ActivationEluLayer& other)
        :Layer(other)
{

}

ActivationEluLayer::ActivationEluLayer(const ActivationEluLayer& other)
        :Layer(other)
{

}

ActivationEluLayer::ActivationEluLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_activation_elu_layer();
}

ActivationEluLayer::~ActivationEluLayer()
{

}

ActivationEluLayer& ActivationEluLayer::operator= (ActivationEluLayer& other)
{
    copy(other);
    return *this;
}

ActivationEluLayer& ActivationEluLayer::operator= (const ActivationEluLayer& other)
{
    copy(other);
    return *this;
}


std::string ActivationEluLayer::asString()
{
    std::string result;

    result+= "ELU\t\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}


void ActivationEluLayer::forward(Tensor &output, Tensor &input)
{
    activation_elu_layer_forward(output, input);
}

void ActivationEluLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias)
{
    (void)input;
    (void)update_weights;
    (void)update_bias;

    activation_elu_layer_backward(error_back, output, error);
}


void ActivationEluLayer::init_activation_elu_layer()
{
    this->m_output_shape            = this->m_input_shape;
    this->m_trainable_parameters    = 0;
    this->m_flops                   = 2*this->m_input_shape.size();
}
