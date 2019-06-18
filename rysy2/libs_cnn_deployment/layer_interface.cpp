#include <layer_interface.h>
#include <iostream>

LayerInterface::LayerInterface()
{
    this->input_shape.w = 0;
    this->input_shape.h = 0;
    this->input_shape.d = 0;

    this->kernel_shape.w = 0;
    this->kernel_shape.h = 0;
    this->kernel_shape.d = 0;

    this->output_shape.w = 0;
    this->output_shape.h = 0;
    this->output_shape.d = 0;
}

LayerInterface::LayerInterface(LayerInterface& other)
{
    copy_interface(other);
}

LayerInterface::LayerInterface(const LayerInterface& other)
{
    copy_interface(other);
}

LayerInterface::LayerInterface(Json::Value json, sShape input_shape)
{
    this->input_shape.w = 0;
    this->input_shape.h = 0;
    this->input_shape.d = 0;

    this->kernel_shape.w = 0;
    this->kernel_shape.h = 0;
    this->kernel_shape.d = 0;

    this->output_shape.w = 0;
    this->output_shape.h = 0;
    this->output_shape.d = 0;

    if ( input_shape.w != 0 && input_shape.h != 0 && input_shape.d != 0)
    {
        this->input_shape.w = input_shape.w;
        this->input_shape.h = input_shape.h;
        this->input_shape.d = input_shape.d;
    }

    this->json = json;
}

LayerInterface::~LayerInterface()
{

}

LayerInterface& LayerInterface::operator= (LayerInterface& other)
{
    copy_interface(other);
    return *this;
}

LayerInterface& LayerInterface::operator= (const LayerInterface& other)
{
    copy_interface(other);
    return *this;
}

void LayerInterface::copy_interface(LayerInterface& other)
{
    this->json = other.json;

    this->input_shape  = other.input_shape;
    this->kernel_shape = other.kernel_shape;
    this->output_shape = other.output_shape;

}

void LayerInterface::copy_interface(const LayerInterface& other)
{
    this->json = other.json;

    this->input_shape  = other.input_shape;
    this->kernel_shape = other.kernel_shape;
    this->output_shape = other.output_shape;
}


sShape LayerInterface::get_input_shape()
{
    return this->input_shape;
}

sShape LayerInterface::get_output_shape()
{
    return this->output_shape;
}

sShape LayerInterface::get_kernel_shape()
{
    return this->kernel_shape;
}


unsigned int LayerInterface::get_input_size()
{
    return this->input_shape.w*this->input_shape.h*this->input_shape.d;
}

unsigned int LayerInterface::get_output_size()
{
    return this->output_shape.w*this->output_shape.h*this->output_shape.d;
}

void LayerInterface::print()
{
    std::cout << "[";
    std::cout << input_shape.w << " ";
    std::cout << input_shape.h << " ";
    std::cout << input_shape.d << " ";
    std::cout << "]";

    std::cout << "[";
    std::cout << kernel_shape.w << " ";
    std::cout << kernel_shape.h << " ";
    std::cout << kernel_shape.d << " ";
    std::cout << "]";

    std::cout << "[";
    std::cout << kernel_shape.w << " ";
    std::cout << kernel_shape.h << " ";
    std::cout << kernel_shape.d << " ";
    std::cout << "]";

    std::cout << "\n";
}

void LayerInterface::forward(float *output, float *input)
{
    (void)output;
    (void)input;
}
