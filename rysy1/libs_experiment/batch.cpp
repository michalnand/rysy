#include <batch.h>
#include <iostream>

Batch::Batch()
{
    this->batch_size  = 0;
    this->current_ptr = 0;
    this->dataset     = nullptr;
}


Batch::Batch(DatasetInterface &dataset, unsigned int batch_size)
{
    init(dataset, batch_size);
}

Batch::~Batch()
{

}

void Batch::init(DatasetInterface &dataset, unsigned int batch_size)
{
    this->dataset       = &dataset;
    this->batch_size    = batch_size;
    this->current_ptr   = 0;

    if (this->batch_size > this->dataset->get_training_size())
    {
        this->batch_size = this->dataset->get_training_size();
    }

    input.resize(size());
    output.resize(size());

    sGeometry input_geometry;
    sGeometry output_geometry;

    input_geometry.w = this->dataset->get_width();
    input_geometry.h = this->dataset->get_height();
    input_geometry.d = this->dataset->get_channels();

    output_geometry.w = 1;
    output_geometry.h = 1;
    output_geometry.d = this->dataset->get_output_size();

    for (unsigned int j = 0; j < size(); j++)
    {
        input[j].init(input_geometry);
        output[j].init(output_geometry);
    }

    fill_new();
}


unsigned int Batch::size()
{
    return this->batch_size;
}

void Batch::next()
{
    this->current_ptr = (this->current_ptr+1)%size();
    if (this->current_ptr == 0)
    {
        fill_new();
    }
}

Tensor& Batch::get_input()
{
    return this->input[this->current_ptr];
}

Tensor& Batch::get_output()
{
    return this->output[this->current_ptr];
}

void Batch::fill_new()
{
    for (unsigned int j = 0; j < size(); j++)
    {
        sDatasetItem item = dataset->get_random_training();
        this->input[j].set_from_host(item.input);
        this->output[j].set_from_host(item.output);
    }
}
