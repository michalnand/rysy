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

LayerSoftmax::LayerSoftmax(Json::Value json, sShape input_shape)
                 :LayerInterface(json, input_shape)
{
    this->input_shape.w = 0;
    this->input_shape.h = 0;
    this->input_shape.d = 0;

    if ( input_shape.w != 0 && input_shape.h != 0 && input_shape.d != 0)
    {
        this->input_shape.w = input_shape.w;
        this->input_shape.h = input_shape.h;
        this->input_shape.d = input_shape.d;
    }
    else
    {
        this->input_shape.w = json["input_shape"][0].asInt();
        this->input_shape.h = json["input_shape"][1].asInt();
        this->input_shape.d = json["input_shape"][2].asInt();
    }

    this->kernel_shape.w = 1;
    this->kernel_shape.h = 1;
    this->kernel_shape.d = 1;

    this->output_shape.w = this->input_shape.w;
    this->output_shape.h = this->input_shape.h;
    this->output_shape.d = this->input_shape.d;

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
    (void)other;
}

void LayerSoftmax::copy(const LayerSoftmax& other)
{
    (void)other;
}


void LayerSoftmax::print()
{
    std::cout << "SOFTMAX ";
    std::cout << "[";
    std::cout << input_shape.w << " ";
    std::cout << input_shape.h << " ";
    std::cout << input_shape.d << " ";
    std::cout << "]";

    std::cout << "[";
    std::cout << output_shape.w << " ";
    std::cout << output_shape.h << " ";
    std::cout << output_shape.d << " ";
    std::cout << "]";

    std::cout << "\n";
}

void LayerSoftmax::forward(float *output, float *input)
{
    softmax_layer_forward(  output, input,
                            this->input_shape);
}
