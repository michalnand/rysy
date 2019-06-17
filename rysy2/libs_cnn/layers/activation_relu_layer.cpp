#include <layers/activation_relu_layer.h>

#include <kernels/activation_relu_layer.cuh>

ActivationReluLayer::ActivationReluLayer()
        :Layer()
{

}

ActivationReluLayer::ActivationReluLayer(ActivationReluLayer& other)
        :Layer(other)
{

}

ActivationReluLayer::ActivationReluLayer(const ActivationReluLayer& other)
        :Layer(other)
{

}

ActivationReluLayer::ActivationReluLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_activation_relu_layer();
}

ActivationReluLayer::~ActivationReluLayer()
{

}

ActivationReluLayer& ActivationReluLayer::operator= (ActivationReluLayer& other)
{
    copy(other);
    return *this;
}

ActivationReluLayer& ActivationReluLayer::operator= (const ActivationReluLayer& other)
{
    copy(other);
    return *this;
}


std::string ActivationReluLayer::asString()
{
    std::string result;

    result+= "RELU\t\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}


void ActivationReluLayer::forward(Tensor &output, Tensor &input)
{
    activation_relu_layer_forward(output, input);
}

void ActivationReluLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights)
{
    (void)input;
    (void)update_weights;

    activation_relu_layer_backward(error_back, output, error);
}


void ActivationReluLayer::init_activation_relu_layer()
{
    this->m_output_shape            = this->m_input_shape;
    this->m_trainable_parameters    = 0;
    this->m_flops                   = 2*this->m_input_shape.size();
}
