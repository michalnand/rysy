#include <model.h>
#include <iostream>

Model::Model()
{
    m_name = "model";
}

Model::Model(const Model &other)
{
    copy_model(other);
}

Model& Model::operator = (const Model &other)
{
    copy_model(other);
    return *this;
}

Model::~Model()
{

}

void Model::copy_model(const Model &other)
{
    m_input_shape  = other.m_input_shape;
    m_output_shape = other.m_output_shape;

    m_models       = other.m_models;
    m_name         = other.m_name;
}

Shape Model::input_shape()
{
    return m_input_shape;
}

Shape Model::output_shape()
{
    return m_output_shape;
}

std::string Model::name()
{
    return m_name;
}


unsigned int Model::get_models_count()
{
    return m_models.size();
}

Model& Model::get_model(unsigned int idx)
{
    return m_models[idx];
}

std::string Model::asString(bool full)
{
    std::string result;
    result+= "name : " + name() + "\n";
    result+= "input_shape : " + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + " " + std::to_string(m_input_shape.t()) + "\n";
    result+= "output_shape : " + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + " " + std::to_string(m_output_shape.t()) + "\n";

    if (full)
    {
        result+= "\n";
        for (unsigned int i = 0; i < m_models.size(); i++)
        {
            result+= "    " + m_models[i].asString();
        }
    }

    return result;
}

void Model::print()
{
    std::cout << asString() << "\n";
}

void Model::print_full()
{
    std::cout << asString(true) << "\n";
}


void Model::forward(Tensor &output, Tensor &input)
{
    (void)output;
    (void)input;
}

void Model::train(Tensor &target_output, Tensor &input)
{
    (void)target_output;
    (void)input;
}

void Model::train_from_error(Tensor &error_back, Tensor &error)
{
    (void)error_back;
    (void)error;
}

Model& Model::add(Model &model)
{
    m_models.push_back(model);
    return m_models[m_models.size() - 1];
}

void Model::compile()
{
    for (unsigned int i = 0; i < m_models.size(); i++)
    {
        m_models[i].compile();
    }
}
