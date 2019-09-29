#include <layer_interface.h>
#include <iostream>

LayerInterface::LayerInterface()
{
    this->input_geometry.w = 0;
    this->input_geometry.h = 0;
    this->input_geometry.d = 0;

    this->kernel_geometry.w = 0;
    this->kernel_geometry.h = 0;
    this->kernel_geometry.d = 0;

    this->output_geometry.w = 0;
    this->output_geometry.h = 0;
    this->output_geometry.d = 0;
} 

LayerInterface::LayerInterface(LayerInterface& other)
{
    copy_interface(other);
}

LayerInterface::LayerInterface(const LayerInterface& other)
{
    copy_interface(other);
}

LayerInterface::LayerInterface(Json::Value json, sGeometry input_geometry)
{
    this->input_geometry.w = 0;
    this->input_geometry.h = 0;
    this->input_geometry.d = 0;

    this->kernel_geometry.w = 0;
    this->kernel_geometry.h = 0;
    this->kernel_geometry.d = 0;

    this->output_geometry.w = 0;
    this->output_geometry.h = 0;
    this->output_geometry.d = 0;

    if ( input_geometry.w != 0 && input_geometry.h != 0 && input_geometry.d != 0)
    {
        this->input_geometry.w = input_geometry.w;
        this->input_geometry.h = input_geometry.h;
        this->input_geometry.d = input_geometry.d;
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

    this->input_geometry  = other.input_geometry;
    this->kernel_geometry = other.kernel_geometry;
    this->output_geometry = other.output_geometry;

}

void LayerInterface::copy_interface(const LayerInterface& other)
{
    this->json = other.json;

    this->input_geometry  = other.input_geometry;
    this->kernel_geometry = other.kernel_geometry;
    this->output_geometry = other.output_geometry;
}


sGeometry LayerInterface::get_input_geometry()
{
    return this->input_geometry;
}

sGeometry LayerInterface::get_output_geometry()
{
    return this->output_geometry;
}

sGeometry LayerInterface::get_kernel_geometry()
{
    return this->kernel_geometry;
}


unsigned int LayerInterface::get_input_size()
{
    return this->input_geometry.w*this->input_geometry.h*this->input_geometry.d;
}

unsigned int LayerInterface::get_output_size()
{
    return this->output_geometry.w*this->output_geometry.h*this->output_geometry.d;
}

void LayerInterface::print()
{
    std::cout << "[";
    std::cout << input_geometry.w << " ";
    std::cout << input_geometry.h << " ";
    std::cout << input_geometry.d << " ";
    std::cout << "]";

    std::cout << "[";
    std::cout << kernel_geometry.w << " ";
    std::cout << kernel_geometry.h << " ";
    std::cout << kernel_geometry.d << " ";
    std::cout << "]";

    std::cout << "[";
    std::cout << kernel_geometry.w << " ";
    std::cout << kernel_geometry.h << " ";
    std::cout << kernel_geometry.d << " ";
    std::cout << "]";

    std::cout << "\n";
}

void LayerInterface::forward(float *output, float *input)
{
    (void)output;
    (void)input;
}
