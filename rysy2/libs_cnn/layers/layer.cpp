#include <layers/layer.h>


Layer::Layer()
{
    this->m_input_shape.set(0, 0, 0);
    this->m_output_shape.set(0, 0, 0);

    this->m_max_time_steps = 1;

    this->m_training_mode = false;
}


Layer::Layer(Layer& other)
{
    copy(other);
}

Layer::Layer(const Layer& other)
{
    copy(other);
}

Layer::Layer(Shape input_shape, Json::Value parameters, unsigned int max_time_steps)
{
    init(input_shape, parameters, max_time_steps);
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
    this->m_max_time_steps  = other.m_max_time_steps;

    this->m_training_mode   = other.m_training_mode;
}

void Layer::copy(const Layer& other)
{
    this->m_input_shape     = other.m_input_shape;
    this->m_output_shape    = other.m_output_shape;

    this->m_parameters      = other.m_parameters;
    this->m_max_time_steps  = other.m_max_time_steps;

    this->m_training_mode   = other.m_training_mode;
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

void Layer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights)
{
    (void)error_back;
    (void)error;
    (void)input;
    (void)output;
    (void)update_weights;
}

void Layer::save(std::string file_name_prefix)
{
    (void)file_name_prefix;
}

void Layer::load(std::string file_name_prefix)
{
    (void)file_name_prefix;
}

void Layer::init(Shape input_shape, Json::Value parameters, unsigned int max_time_steps)
{
    this->m_input_shape = input_shape;
    this->m_output_shape.set(1, 1, 1);

    this->m_parameters = parameters;
    this->m_max_time_steps = max_time_steps;

    this->m_training_mode = false;
}
