#include <layers/layer.h>
#include <iostream>

Layer::Layer()
{
    this->m_input_shape.set(0, 0, 0);
    this->m_output_shape.set(0, 0, 0);

    this->m_training_mode = false;

    this->m_flops = 0;
    this->m_trainable_parameters = 0;
    this->time_step_idx = 0;
}


Layer::Layer(Layer& other)
{
    copy(other);
}

Layer::Layer(const Layer& other)
{
    copy(other);
}

Layer::Layer(Shape input_shape, Json::Value parameters)
{
    init(input_shape, parameters);
}

Layer::~Layer()
{

}

Layer& Layer::operator= (Layer& other)
{
    copy(other);
    return *this;
}

Layer& Layer::operator= (const Layer& other)
{
    copy(other);
    return *this;
}


void Layer::copy(Layer& other)
{
    this->m_input_shape     = other.m_input_shape;
    this->m_output_shape    = other.m_output_shape;

    this->m_parameters      = other.m_parameters;
    this->m_training_mode   = other.m_training_mode;

    this->m_flops                   = other.m_flops;
    this->m_trainable_parameters    =  other.m_trainable_parameters;

    this->time_step_idx = other.time_step_idx;
}

void Layer::copy(const Layer& other)
{
    this->m_input_shape     = other.m_input_shape;
    this->m_output_shape    = other.m_output_shape;

    this->m_parameters      = other.m_parameters;
    this->m_training_mode   = other.m_training_mode;

    this->m_flops           = other.m_flops;
    this->m_trainable_parameters      =  other.m_trainable_parameters;

    this->time_step_idx = other.time_step_idx;
}


Shape Layer::get_input_shape()
{
    return this->m_input_shape;
}

Shape Layer::get_output_shape()
{
    return this->m_output_shape;
}


void Layer::reset()
{

}

void Layer::set_training_mode()
{
    m_training_mode = true;
}

void Layer::unset_training_mode()
{
    m_training_mode = false;
}

void Layer::forward(Tensor &output, Tensor &input)
{
    (void)output;
    (void)input;
}

void Layer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias)
{
    (void)error_back;
    (void)error;
    (void)input;
    (void)output;
    (void)update_weights;
    (void)update_bias;
}

void Layer::save(std::string file_name_prefix)
{
    (void)file_name_prefix;
}

void Layer::load(std::string file_name_prefix)
{
    (void)file_name_prefix;
}

std::string Layer::asString()
{
    std::string result;

    result = "[INTERFACE]";

    return result;
}

unsigned long int Layer::get_flops()
{
    return this->m_flops;
}

unsigned long int Layer::get_trainable_parameters()
{
    return this->m_trainable_parameters;
}

void Layer::set_time_step(unsigned int time_step_idx)
{
    this->time_step_idx = time_step_idx;
}

void Layer::init(Shape input_shape, Json::Value parameters)
{
    this->m_input_shape = input_shape;
    this->m_output_shape.set(1, 1, 1);

    this->m_parameters = parameters;
    this->m_training_mode = false;

    this->m_flops = 0;
    this->m_trainable_parameters = 0;

    this->time_step_idx = 0;
}
