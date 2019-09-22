#include <layers/softmax_layer.h>

#include <kernels/softmax_layer.cuh>

SoftmaxLayer::SoftmaxLayer()
        :Layer()
{

}

SoftmaxLayer::SoftmaxLayer(SoftmaxLayer& other)
        :Layer(other)
{

}

SoftmaxLayer::SoftmaxLayer(const SoftmaxLayer& other)
        :Layer(other)
{

}

SoftmaxLayer::SoftmaxLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_activation_elu_layer();
}

SoftmaxLayer::~SoftmaxLayer()
{

}

SoftmaxLayer& SoftmaxLayer::operator= (SoftmaxLayer& other)
{
    copy(other);
    return *this;
}

SoftmaxLayer& SoftmaxLayer::operator= (const SoftmaxLayer& other)
{
    copy(other);
    return *this;
}


std::string SoftmaxLayer::asString()
{
    std::string result;

    result+= "SOFTMAX\t\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}


void SoftmaxLayer::forward(Tensor &output, Tensor &input)
{
    softmax_layer_forward(output, input);
}

void SoftmaxLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias)
{
    (void)input;
    (void)update_weights;
    (void)update_bias;

    softmax_layer_backward(error_back, output, error);
}


void SoftmaxLayer::init_activation_elu_layer()
{
    this->m_output_shape            = this->m_input_shape;
    this->m_trainable_parameters    = 0;
    this->m_flops                   = 10*this->m_input_shape.size();
}
