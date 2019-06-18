#include <layer_max_pooling.h>

#include <kernels/max_pooling_layer_forward.cuh>

#include <iostream>

LayerMaxPooling::LayerMaxPooling()
                 :LayerInterface()
{

}

LayerMaxPooling::LayerMaxPooling(LayerMaxPooling& other)
                 :LayerInterface(other)
{
    copy(other);
}

LayerMaxPooling::LayerMaxPooling(const LayerMaxPooling& other)
                 :LayerInterface(other)
{
    copy(other);
}

LayerMaxPooling::LayerMaxPooling(Json::Value json, sShape input_shape)
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

    this->kernel_shape.w = json["shape"][0].asInt();
    this->kernel_shape.h = json["shape"][1].asInt();
    this->kernel_shape.d = json["shape"][2].asInt();

    this->output_shape.w = this->input_shape.w/2;
    this->output_shape.h = this->input_shape.h/2;
    this->output_shape.d = this->input_shape.d;

    this->json = json;
}

LayerMaxPooling::~LayerMaxPooling()
{

}

LayerMaxPooling& LayerMaxPooling::operator= (LayerMaxPooling& other)
{
    copy_interface(other);
    copy(other);

    return *this;
}

LayerMaxPooling& LayerMaxPooling::operator= (const LayerMaxPooling& other)
{
    copy_interface(other);
    copy(other);

    return *this;
}

void LayerMaxPooling::copy(LayerMaxPooling& other)
{
    (void)other;
}

void LayerMaxPooling::copy(const LayerMaxPooling& other)
{
    (void)other;
}

void LayerMaxPooling::print()
{
    std::cout << "MAX POOLING ";
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
    std::cout << output_shape.w << " ";
    std::cout << output_shape.h << " ";
    std::cout << output_shape.d << " ";
    std::cout << "]";

    std::cout << "\n";
}


void LayerMaxPooling::forward(float *output, float *input)
{
    max_pooling_layer_forward(  output, input,
                                this->output_shape );
}
