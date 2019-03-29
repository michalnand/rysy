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

LayerMaxPooling::LayerMaxPooling(Json::Value json, sGeometry input_geometry)
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

    this->kernel_geometry.w = json["geometry"][0].asInt();
    this->kernel_geometry.h = json["geometry"][1].asInt();
    this->kernel_geometry.d = json["geometry"][2].asInt();

    this->output_geometry.w = this->input_geometry.w/2;
    this->output_geometry.h = this->input_geometry.h/2;
    this->output_geometry.d = this->input_geometry.d;

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
    std::cout << output_geometry.w << " ";
    std::cout << output_geometry.h << " ";
    std::cout << output_geometry.d << " ";
    std::cout << "]";

    std::cout << "\n";
}


void LayerMaxPooling::forward(float *output, float *input)
{
    return;
    
    max_pooling_layer_forward(  output, input,
                                this->output_geometry );
}
