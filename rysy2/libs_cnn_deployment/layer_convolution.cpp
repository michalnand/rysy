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

LayerConvolution::LayerConvolution(Json::Value json, sShape input_shape)
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

    this->output_shape.w = this->input_shape.w;
    this->output_shape.h = this->input_shape.h;
    this->output_shape.d = this->kernel_shape.d;

    this->json = json;

    std::string weights_file_name   = this->json["weights_file_name"].asString() + "_weight.bin";
    std::string bias_file_name      = this->json["weights_file_name"].asString() + "_bias.bin";

    unsigned int weights_size   = this->kernel_shape.w*this->kernel_shape.h*this->kernel_shape.d*this->input_shape.d;
    unsigned int bias_size      = this->output_shape.d;


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
    unsigned int weights_size   = this->kernel_shape.w*this->kernel_shape.d*this->kernel_shape.d*this->input_shape.d;
    unsigned int bias_size      = this->output_shape.d;

    cuda_float_allocator.device_to_device(this->weights, other.weights, weights_size);
    cuda_float_allocator.device_to_device(this->bias, other.bias, bias_size);
}

void LayerConvolution::copy(const LayerConvolution& other)
{
    unsigned int weights_size   = this->kernel_shape.w*this->kernel_shape.d*this->kernel_shape.d*this->input_shape.d;
    unsigned int bias_size      = this->output_shape.d;

    cuda_float_allocator.device_to_device(this->weights, other.weights, weights_size);
    cuda_float_allocator.device_to_device(this->bias, other.bias, bias_size);
}


void LayerConvolution::print()
{
    std::cout << "CONVOLUTION ";
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

void LayerConvolution::forward(float *output, float *input)
{
    convolution_layer_forward(  output, input,
                                this->weights, this->bias,
                                this->input_shape,
                                this->kernel_shape,
                                this->output_shape );
}