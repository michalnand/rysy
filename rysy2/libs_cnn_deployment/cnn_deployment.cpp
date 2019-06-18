#include <cnn_deployment.h>

#include <cuda_float_allocator.cuh>

#include <layer_convolution.h>
#include <layer_max_pooling.h>
#include <layer_output.h>
#include <layer_softmax.h>
#include <timer.h>


#include <iostream>

CNNDeployment::CNNDeployment()
{
    buffer_a = nullptr;
    buffer_b = nullptr;

    input_shape.w = 0;
    input_shape.h = 0;
    input_shape.d = 0;

    output_shape.w = 0;
    output_shape.h = 0;
    output_shape.d = 0;
}

CNNDeployment::CNNDeployment(CNNDeployment& other)
{
    copy(other);
}

CNNDeployment::CNNDeployment(const CNNDeployment& other)
{
    copy(other);
}

CNNDeployment::CNNDeployment(std::string json_file_name, sShape input_shape)
{
    init(json_file_name, input_shape);
}

CNNDeployment::~CNNDeployment()
{
    if (buffer_a != nullptr)
        cuda_float_allocator.free(buffer_a);

    if (buffer_b != nullptr)
        cuda_float_allocator.free(buffer_b);

    for (unsigned int i = 0; i < layers.size(); i++)
        delete layers[i];
}

CNNDeployment& CNNDeployment::operator= (CNNDeployment& other)
{
    copy(other);
    return *this;
}

CNNDeployment& CNNDeployment::operator= (const CNNDeployment& other)
{
    copy(other);
    return *this;
}

void CNNDeployment::copy(CNNDeployment& other)
{

}

void CNNDeployment::copy(const CNNDeployment& other)
{

}




void CNNDeployment::init(std::string json_file_name, sShape input_shape)
{
    JsonConfig json(json_file_name);
    auto json_config = json.result;

    output_shape.w = 10;
    output_shape.h = 10;
    output_shape.d = 3;

    if ((input_shape.w == 0) ||
        (input_shape.h == 0) ||
        (input_shape.d == 0) )
    {
        this->input_shape.w  = json_config["input_shape"][0].asInt();
        this->input_shape.h  = json_config["input_shape"][1].asInt();
        this->input_shape.d  = json_config["input_shape"][2].asInt();
    }
    else
    {
      this->input_shape.w  = input_shape.w;
      this->input_shape.h  = input_shape.h;
      this->input_shape.d  = input_shape.d;
    }

    sShape layer_input_shape    = input_shape;

    for (auto layer: json_config["layers"])
    {
        LayerInterface *layer_ = create_layer(layer, layer_input_shape);

        if (layer_ != nullptr)
        {
            layer_->print();
            layers.push_back(layer_);
            layer_input_shape = layer_->get_output_shape();

            if (layer["type"].asString() == "output")
            {
                LayerInterface *layer_ = new LayerSoftmax(layer, layer_input_shape);

                layer_->print();
                layers.push_back(layer_);
                layer_input_shape = layer_->get_output_shape();
            }
        }
    }

    unsigned int buffer_size = get_input_size();


    for (unsigned int i = 0; i < layers.size(); i++)
    {
        if (layers[i]->get_input_size() > buffer_size)
            buffer_size = layers[i]->get_input_size();

        if (layers[i]->get_output_size() > buffer_size)
            buffer_size = layers[i]->get_output_size();
    }

    std::cout << "requested buffer size " << (buffer_size*2*sizeof(float))/1000000 << "MB\n";

    buffer_a = cuda_float_allocator.malloc(buffer_size);
    buffer_b = cuda_float_allocator.malloc(buffer_size);
}

sShape CNNDeployment::get_input_shape()
{
    return input_shape;
}

sShape CNNDeployment::get_output_shape()
{
    return output_shape;
}

unsigned int CNNDeployment::get_input_size()
{
    return input_shape.w*input_shape.h*input_shape.d;
}

unsigned int CNNDeployment::get_output_size()
{
    return output_shape.w*output_shape.h*output_shape.d;
}


void CNNDeployment::forward(std::vector<float> &output, std::vector<float> &input)
{
    float *input_buffer  = buffer_a;
    float *output_buffer = buffer_b;
    float *tmp_buffer    = nullptr;

    cuda_float_allocator.host_to_device(input_buffer, &input[0], get_input_size());

    for (unsigned int i = 0; i < layers.size(); i++)
    {
        layers[i]->forward(output_buffer, input_buffer);

        tmp_buffer = output_buffer;
        output_buffer = input_buffer;
        input_buffer = tmp_buffer;
    }

    cuda_float_allocator.device_to_host(&output[0], input_buffer, get_output_size());
}


LayerInterface* CNNDeployment::create_layer(Json::Value json, sShape input_shape)
{
    LayerInterface *result = nullptr;

    std::string type = json["type"].asString();

    if (type == "convolution")
        result = new LayerConvolution(json, input_shape);

    if ( (type == "max pooling") || (type == "max_pooling"))
        result = new LayerMaxPooling(json, input_shape);

    if (type == "output")
    {
        result = new LayerOutput(json, input_shape);

        this->output_shape.w = result->get_output_shape().w;
        this->output_shape.h = result->get_output_shape().h;
        this->output_shape.d = result->get_output_shape().d;
    }


    return result;
}
