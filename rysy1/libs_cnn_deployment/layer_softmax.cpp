#include <layer_softmax.h>
#include <cuda_float_allocator.cuh>
#include <kernels/softmax_layer_forward.cuh>

#include <iostream>

LayerSoftmax::LayerSoftmax()
                 :LayerInterface()
{

}

LayerSoftmax::LayerSoftmax(LayerSoftmax& other)
                 :LayerInterface(other)
{
    copy(other);
}

LayerSoftmax::LayerSoftmax(const LayerSoftmax& other)
                 :LayerInterface(other)
{
    copy(other);
}
 
LayerSoftmax::LayerSoftmax(Json::Value json, sGeometry input_geometry)
                 :LayerInterface(json, input_geometry)
{
    this->input_geometry.w = 0;
    this->input_geometry.h = 0;
    this->input_geometry.d = 0;

    if ( input_geometry.w != 0 && input_geometry.h != 0 && input_geometry.d != 0)
    {
        this->input_geometry.w = input_geometry.w;
        this->input_geometry.h = input_geometry.h;
        this->input_geometry.d = input_geometry.d;
    }
    else
    {
        this->input_geometry.w = json["input_geometry"][0].asInt();
        this->input_geometry.h = json["input_geometry"][1].asInt();
        this->input_geometry.d = json["input_geometry"][2].asInt();
    }

    this->kernel_geometry.w = 1;
    this->kernel_geometry.h = 1;
    this->kernel_geometry.d = 1;

    this->output_geometry.w = this->input_geometry.w;
    this->output_geometry.h = this->input_geometry.h;
    this->output_geometry.d = this->input_geometry.d;

    this->json = json;
}

LayerSoftmax::~LayerSoftmax()
{

}

LayerSoftmax& LayerSoftmax::operator= (LayerSoftmax& other)
{
    copy_interface(other);
    copy(other);

    return *this;
}

LayerSoftmax& LayerSoftmax::operator= (const LayerSoftmax& other)
{
    copy_interface(other);
    copy(other);

    return *this;
}

void LayerSoftmax::copy(LayerSoftmax& other)
{

}

void LayerSoftmax::copy(const LayerSoftmax& other)
{

}


void LayerSoftmax::print()
{
    std::cout << "SOFTMAX ";
    std::cout << "[";
    std::cout << input_geometry.w << " ";
    std::cout << input_geometry.h << " ";
    std::cout << input_geometry.d << " ";
    std::cout << "]";

    std::cout << "[";
    std::cout << output_geometry.w << " ";
    std::cout << output_geometry.h << " ";
    std::cout << output_geometry.d << " ";
    std::cout << "]";

    std::cout << "\n";
}

void LayerSoftmax::forward(float *output, float *input)
{
    softmax_layer_forward(  output, input,
                            this->input_geometry);
}
