#include <layer_convolution.h>
#include <cuda_float_allocator.cuh>
#include <kernels/convolution_layer_forward.cuh>

#include <iostream>

LayerConvolution::LayerConvolution()
                 :LayerInterface()
{
    weights = nullptr;
    bias    = nullptr;
}

LayerConvolution::LayerConvolution(LayerConvolution& other)
                 :LayerInterface(other)
{
    copy(other);
} 

LayerConvolution::LayerConvolution(const LayerConvolution& other)
                 :LayerInterface(other)
{
    copy(other);
}

LayerConvolution::LayerConvolution(Json::Value json, sGeometry input_geometry)
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

    this->output_geometry.w = this->input_geometry.w;
    this->output_geometry.h = this->input_geometry.h;
    this->output_geometry.d = this->kernel_geometry.d;

    this->json = json;

    std::string weights_file_name   = this->json["weights_file_name"].asString() + "_weight.bin";
    std::string bias_file_name      = this->json["weights_file_name"].asString() + "_bias.bin";

    unsigned int weights_size   = this->kernel_geometry.w*this->kernel_geometry.h*this->kernel_geometry.d*this->input_geometry.d;
    unsigned int bias_size      = this->output_geometry.d;


    this->weights = cuda_float_allocator.malloc(weights_size);
    this->bias    = cuda_float_allocator.malloc(bias_size);

    cuda_float_allocator.load_from_file(this->weights, weights_file_name, weights_size);
    cuda_float_allocator.load_from_file(this->bias, bias_file_name, bias_size);
}

LayerConvolution::~LayerConvolution()
{
    if (this->weights != nullptr)
        cuda_float_allocator.free(this->weights);

    if (this->bias != nullptr)
        cuda_float_allocator.free(this->bias);
}

LayerConvolution& LayerConvolution::operator= (LayerConvolution& other)
{
    copy_interface(other);
    copy(other);

    return *this;
}

LayerConvolution& LayerConvolution::operator= (const LayerConvolution& other)
{
    copy_interface(other);
    copy(other);

    return *this;
}

void LayerConvolution::copy(LayerConvolution& other)
{
    unsigned int weights_size   = this->kernel_geometry.w*this->kernel_geometry.d*this->kernel_geometry.d*this->input_geometry.d;
    unsigned int bias_size      = this->output_geometry.d;

    cuda_float_allocator.device_to_device(this->weights, other.weights, weights_size);
    cuda_float_allocator.device_to_device(this->bias, other.bias, bias_size);
}

void LayerConvolution::copy(const LayerConvolution& other)
{
    unsigned int weights_size   = this->kernel_geometry.w*this->kernel_geometry.d*this->kernel_geometry.d*this->input_geometry.d;
    unsigned int bias_size      = this->output_geometry.d;

    cuda_float_allocator.device_to_device(this->weights, other.weights, weights_size);
    cuda_float_allocator.device_to_device(this->bias, other.bias, bias_size);
}


void LayerConvolution::print()
{
    std::cout << "CONVOLUTION ";
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

void LayerConvolution::forward(float *output, float *input)
{
    convolution_layer_forward(  output, input,
                                this->weights, this->bias,
                                this->input_geometry,
                                this->kernel_geometry,
                                this->output_geometry );
}
