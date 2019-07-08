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
    std::cout << "input_shape  = " << input_shape.w() << " " << input_shape.h() << " " << input_shape.d() << " " << input_shape.t() << "\n";
    std::cout << "output_shape = " << output_shape.w() << " " << output_shape.h() << " " << output_shape.d() << " " << output_shape.t() << "\n";
    std::cout << "training_count = " << get_training_count() << "\n";
    std::cout << "testing_count  = " << get_testing_count() << "\n";
}

void DatasetInterface::clear()
{
    for (unsigned int i = 0; i < training_input.size(); i++)
        training_input[i].clear();
    training_input.clear();

    for (unsigned int i = 0; i < training_output.size(); i++)
        training_output[i].clear();
    training_output.clear();


    for (unsigned int i = 0; i < testing_input.size(); i++)
        testing_input[i].clear();
    testing_input.clear();

    for (unsigned int i = 0; i < testing_output.size(); i++)
        testing_output[i].clear();
    testing_output.clear();


    this->input_shape.set(0, 0, 0);
    this->output_shape.set(0, 0, 0);
}

void DatasetInterface::normalise_mat(std::vector<std::vector<float>> &mat)
{
    float max = mat[0][0];
    float min = max;

    for (unsigned int j = 0; j < mat.size(); j++)
        for (unsigned int i = 0; i < mat[j].size(); i++)
        {
            if (mat[j][i] > max)
                max = mat[j][i];

            if (mat[j][i] < min)
                min = mat[j][i];
        }

    float k = 0.0;
    float q = 0.0;

    if (max > min)
    {
        k = (1.0 - 0.0)/(max - min);
        q = 1.0 - k*max;
    }

    for (unsigned int j = 0; j < mat.size(); j++)
        for (unsigned int i = 0; i < mat[j].size(); i++)
            mat[j][i] = k*mat[j][i] + q;
}

void DatasetInterface::normalise_input()
{
    normalise_mat(training_input);
    normalise_mat(testing_input);
}

void DatasetInterface::normalise_output()
{
    normalise_mat(training_output);
    normalise_mat(testing_output);
}
