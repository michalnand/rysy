#include <batch.h>

Batch::Batch(Shape input_shape, Shape output_shape, unsigned int batch_size)
{
    this->input.resize(batch_size);
    for (unsigned int i = 0; i < batch_size; i++)
    {
        this->input[i].init(input_shape);
    }

    this->output.resize(batch_size);
    for (unsigned int i = 0; i < batch_size; i++)
    {
        this->output[i].init(output_shape);
    }
}

Batch::Batch(Batch& other)
{
    copy(other);
}

Batch::Batch(const Batch& other)
{
    copy(other);
}

Batch::~Batch()
{

}

Batch& Batch::operator= (Batch& other)
{
    copy(other);
    return *this;
}

Batch& Batch::operator= (const Batch& other)
{
    copy(other);
    return *this;
}


void Batch::copy(Batch& other)
{
    this->input     = other.input;
    this->output    = other.output;
}

void Batch::copy(const Batch& other)
{
    this->input     = other.input;
    this->output    = other.output;
}


void Batch::create(DatasetInterface &dataset)
{
    for (unsigned int i = 0; i < this->input.size(); i++)
    {
        dataset.set_random_training_idx();
        this->input[i].set_from_host(dataset.get_training_input());
        this->output[i].set_from_host(dataset.get_training_output());
    }
}

std::vector<Tensor>& Batch::get_input_all()
{
    return this->input;
}

std::vector<Tensor>& Batch::get_output_all()
{
    return this->output;
}
