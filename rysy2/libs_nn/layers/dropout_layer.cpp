#include <layers/dropout_layer.h>

#include <kernels/dropout_layer.cuh>

DropoutLayer::DropoutLayer()
        :Layer()
{
    this->m_dropout_level = 0.0;
}

DropoutLayer::DropoutLayer(DropoutLayer& other)
        :Layer(other)
{
    copy_dropout(other);
}

DropoutLayer::DropoutLayer(const DropoutLayer& other)
        :Layer(other)
{
    copy_dropout(other);
}

DropoutLayer::DropoutLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_dropout_layer();
}

DropoutLayer::~DropoutLayer()
{

}

DropoutLayer& DropoutLayer::operator= (DropoutLayer& other)
{
    copy(other);
    copy_dropout(other);
    return *this;
}

DropoutLayer& DropoutLayer::operator= (const DropoutLayer& other)
{
    copy(other);
    copy_dropout(other);
    return *this;
}


std::string DropoutLayer::asString()
{
    std::string result;

    result+= "DROPOUT\t\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}

void DropoutLayer::copy_dropout(DropoutLayer &other)
{
    this->m_dropout_level = other.m_dropout_level;
    this->noise = other.noise;
}

void DropoutLayer::copy_dropout(const DropoutLayer &other)
{
    this->m_dropout_level = other.m_dropout_level;
    this->noise = other.noise;
}

void DropoutLayer::forward(Tensor &output, Tensor &input)
{
    if (m_training_mode)
    {
        noise.set_random(1.0);
        dropout_layer_forward(output, input, noise, m_dropout_level);
    }
    else
    {
        output = input;
        output.mul(1.0 - m_dropout_level);
    }
}

void DropoutLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias)
{
    (void)input;
    (void)update_weights;
    (void)update_bias;

    dropout_layer_backward(error_back, output, error);
}


void DropoutLayer::init_dropout_layer()
{
    this->m_output_shape            = this->m_input_shape;
    this->m_trainable_parameters    = 0;
    this->m_flops                   = 2*this->m_input_shape.size();

    this->m_dropout_level = m_parameters["hyperparameters"]["dropout"].asFloat();

    noise.init(m_output_shape);
    noise.set_random(1.0);
}
