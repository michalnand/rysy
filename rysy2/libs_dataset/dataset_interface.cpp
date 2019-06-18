#include <dataset_interface.h>
#include <iostream>

DatasetInterface::DatasetInterface()
{
    srand(time(NULL));
    current_training_idx = 0;
}

DatasetInterface::DatasetInterface(DatasetInterface& other)
{
    srand(time(NULL));
    copy(other);
}

DatasetInterface::DatasetInterface(const DatasetInterface& other)
{
    srand(time(NULL));
    copy(other);
}

DatasetInterface::DatasetInterface(Shape input_shape, Shape output_shape)
{
    srand(time(NULL));
    current_training_idx = 0;

    this->input_shape   = input_shape;
    this->output_shape  = output_shape;
}

DatasetInterface::~DatasetInterface()
{

}

DatasetInterface& DatasetInterface::operator= (DatasetInterface& other)
{
    copy(other);
    return *this;
}

DatasetInterface& DatasetInterface::operator= (const DatasetInterface& other)
{
    copy(other);
    return *this;
}


void DatasetInterface::copy(DatasetInterface& other)
{
    this->input_shape           = other.input_shape;
    this->output_shape          = other.output_shape;
    this->current_training_idx  = other.current_training_idx;

    this->training_input    = other.training_input;
    this->training_output   = other.training_output;

    this->testing_input    = other.testing_input;
    this->testing_output   = other.testing_output;
}

void DatasetInterface::copy(const DatasetInterface& other)
{
    this->input_shape           = other.input_shape;
    this->output_shape          = other.output_shape;
    this->current_training_idx  = other.current_training_idx;

    this->training_input    = other.training_input;
    this->training_output   = other.training_output;

    this->testing_input    = other.testing_input;
    this->testing_output   = other.testing_output;
}



Shape DatasetInterface::get_input_shape()
{
    return this->input_shape;
}

Shape DatasetInterface::get_output_shape()
{
    return this->output_shape;
}

unsigned int DatasetInterface::get_training_count()
{
    return this->training_output.size();
}

unsigned int DatasetInterface::get_testing_count()
{
    return this->testing_output.size();
}

unsigned int DatasetInterface::get_classes_count()
{
    return this->output_shape.size();
}

void DatasetInterface::set_training_idx(unsigned int idx)
{
    this->current_training_idx = idx%get_training_count();
}

void DatasetInterface::set_random_training_idx()
{
    this->current_training_idx = rand()%get_training_count();
}

std::vector<float>& DatasetInterface::get_training_input()
{
    return this->training_input[this->current_training_idx];
}

std::vector<float>& DatasetInterface::get_training_output()
{
    return this->training_output[this->current_training_idx];
}

std::vector<std::vector<float>>& DatasetInterface::get_training_input_all()
{
    return this->training_input;
}

std::vector<std::vector<float>>& DatasetInterface::get_training_output_all()
{
    return this->training_output;
}

std::vector<float>& DatasetInterface::get_testing_input(unsigned int idx)
{
    return this->testing_input[idx];
}

std::vector<float>& DatasetInterface::get_testing_output(unsigned int idx)
{
    return this->testing_output[idx];
}

std::vector<std::vector<float>>& DatasetInterface::get_testing_input_all()
{
    return this->testing_input;
}

std::vector<std::vector<float>>& DatasetInterface::get_testing_output_all()
{
    return this->testing_output;
}


void DatasetInterface::set_input_shape(Shape input_shape)
{
    this->input_shape = input_shape;
}

void DatasetInterface::set_output_shape(Shape output_shape)
{
    this->output_shape = output_shape;
}

void DatasetInterface::add_training(std::vector<float>& input, std::vector<float> &output)
{
    this->training_input.push_back(input);
    this->training_output.push_back(output);
}

void DatasetInterface::add_testing(std::vector<float>& input, std::vector<float> &output)
{
    this->testing_input.push_back(input);
    this->testing_output.push_back(output);
}

void DatasetInterface::print()
{
    std::cout << "input_shape  = " << input_shape.w() << " " << input_shape.h() << " " << input_shape.d() << "\n";
    std::cout << "output_shape = " << output_shape.w() << " " << output_shape.h() << " " << output_shape.d() << "\n";
    std::cout << "training_count = " << get_training_count() << "\n";
    std::cout << "testing_count  = " << get_testing_count() << "\n";
}