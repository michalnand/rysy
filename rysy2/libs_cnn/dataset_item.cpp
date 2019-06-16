#include <dataset_item.h>
#include <iostream>

DatasetItem::DatasetItem()
{

}

DatasetItem::DatasetItem(DatasetItem& other)
{
    copy(other);
}

DatasetItem::DatasetItem(const DatasetItem& other)
{
    copy(other);
}

DatasetItem::DatasetItem(Shape input_shape, Shape output_shape,
                         unsigned int input_time_steps, unsigned int output_time_steps)
{
    init(input_shape, output_shape, input_time_steps, output_time_steps);
}

DatasetItem::DatasetItem(    unsigned int input_width, unsigned int input_height, unsigned int input_depth,
                             unsigned int output_width, unsigned int output_height, unsigned int output_depth,
                             unsigned int input_time_steps,
                             unsigned int output_time_steps)
{
    Shape input_shape(input_width, input_height, input_depth);
    Shape output_shape(output_width, output_height, output_depth);

    init(input_shape, output_shape, input_time_steps, output_time_steps);
}


DatasetItem::~DatasetItem()
{

}

DatasetItem& DatasetItem::operator= (DatasetItem& other)
{
    copy(other);
    return *this;
}

DatasetItem& DatasetItem::operator= (const DatasetItem& other)
{
    copy(other);
    return *this;
}

void DatasetItem::copy(DatasetItem& other)
{
    this->input_shape   = other.input_shape;
    this->output_shape  = other.output_shape;
    this->input         = other.input;
    this->output        = other.output;
}

void DatasetItem::copy(const DatasetItem& other)
{
    this->input_shape   = other.input_shape;
    this->output_shape  = other.output_shape;
    this->input         = other.input;
    this->output        = other.output;
}


void DatasetItem::init(  Shape input_shape,
                         Shape output_shape,
                         unsigned int input_time_steps, unsigned int output_time_steps)
{
    this->input_shape  = input_shape;
    this->output_shape = output_shape;

    for (unsigned int i = 0; i < input_time_steps; i++)
    {
        this->input.push_back(Tensor(this->input_shape));
    }

    for (unsigned int i = 0; i < output_time_steps; i++)
    {
        this->output.push_back(Tensor(this->output_shape));
    }
}

unsigned int DatasetItem::get_input_time_steps()
{
    return this->input.size();
}

unsigned int DatasetItem::get_output_time_steps()
{
    return this->output.size();
}

Shape DatasetItem::get_input_shape()
{
    return this->input_shape;
}

Shape DatasetItem::get_output_shape()
{
    return this->output_shape;
}

Tensor& DatasetItem::get_input(unsigned int time_step)
{
    return this->input[time_step];
}

Tensor& DatasetItem::get_output(unsigned int time_step)
{
    return this->output[time_step];
}


void DatasetItem::set_input(std::vector<std::vector<float>> &input)
{
    for (unsigned int i = 0; i < get_input_time_steps(); i++)
        this->input[i].set_from_host(input[i]);
}

void DatasetItem::set_input(std::vector<float> &input, unsigned int time_step)
{
    this->input[time_step].set_from_host(input);
}

void DatasetItem::set_output(std::vector<std::vector<float>> &output)
{
    for (unsigned int i = 0; i < get_output_time_steps(); i++)
        this->output[i].set_from_host(output[i]);
}

void DatasetItem::set_output(std::vector<float> &output, unsigned int time_step)
{
    this->output[time_step].set_from_host(output);
}

void DatasetItem::print(bool full)
{
    std::cout << "input shape "; this->input_shape.print(); std::cout << "\n";
    std::cout << "output shape "; this->output_shape.print(); std::cout << "\n";
    std::cout << "input time steps " << get_input_time_steps() << "\n";
    std::cout << "output time steps " << get_output_time_steps() << "\n";

    std::cout << "\n";

    if (full)
    {
        std::cout << "\n";
        std::cout << "INPUT : \n";
        for (unsigned int i = 0; i < get_input_time_steps(); i++)
            this->input[i].print();

        std::cout << "\n\n";
        std::cout << "OUTPUT : \n";
        for (unsigned int i = 0; i < get_output_time_steps(); i++)
            this->output[i].print();

        std::cout << "\n\n\n\n\n";
    }
}

void DatasetItem::_set_random()
{
    for (unsigned int i = 0; i < get_input_time_steps(); i++)
        this->input[i].set_random(1.0);

    for (unsigned int i = 0; i < get_output_time_steps(); i++)
        this->output[i].set_random(1.0);
}
