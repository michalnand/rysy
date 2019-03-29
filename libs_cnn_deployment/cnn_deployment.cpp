#include <cnn_deployment.h>

#include <cuda_float_allocator.cuh>

#include <layer_convolution.h>
#include <layer_max_pooling.h>
#include <layer_output.h>

#include <iostream>

CNNDeployment::CNNDeployment()
{
    buffer_a = nullptr;
    buffer_b = nullptr;

    input_geometry.w = 0;
    input_geometry.h = 0;
    input_geometry.d = 0;

    output_geometry.w = 0;
    output_geometry.h = 0;
    output_geometry.d = 0;
}

CNNDeployment::CNNDeployment(CNNDeployment& other)
{
    copy(other);
}

CNNDeployment::CNNDeployment(const CNNDeployment& other)
{
    copy(other);
}

CNNDeployment::CNNDeployment(std::string json_file_name, sGeometry input_geometry)
{
    init(json_file_name, input_geometry);
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




void CNNDeployment::init(std::string json_file_name, sGeometry input_geometry)
{
    JsonConfig json(json_file_name);
    auto json_config = json.result;

    output_geometry.w = 10;
    output_geometry.h = 10;
    output_geometry.d = 3;

    if ((input_geometry.w == 0) ||
        (input_geometry.h == 0) ||
        (input_geometry.d == 0) )
    {
        this->input_geometry.w  = json_config["input_geometry"][0].asInt();
        this->input_geometry.h  = json_config["input_geometry"][1].asInt();
        this->input_geometry.d  = json_config["input_geometry"][2].asInt();
    }
    else
    {
      this->input_geometry.w  = input_geometry.w;
      this->input_geometry.h  = input_geometry.h;
      this->input_geometry.d  = input_geometry.d;
    }

    sGeometry layer_input_geometry    = input_geometry;

    for (auto layer: json_config["layers"])
    {
        LayerInterface *layer_ = create_layer(layer, layer_input_geometry);

        if (layer_ != nullptr)
        {
            layer_->print();
            layers.push_back(layer_);
            layer_input_geometry = layer_->get_output_geometry();
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

sGeometry CNNDeployment::get_input_geometry()
{
    return input_geometry;
}

sGeometry CNNDeployment::get_output_geometry()
{
    return output_geometry;
}

unsigned int CNNDeployment::get_input_size()
{
    return input_geometry.w*input_geometry.h*input_geometry.d;
}

unsigned int CNNDeployment::get_output_size()
{
    return output_geometry.w*output_geometry.h*output_geometry.d;
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


LayerInterface* CNNDeployment::create_layer(Json::Value json, sGeometry input_geometry)
{
    LayerInterface *result = nullptr;

    std::string type = json["type"].asString();

    if (type == "convolution")
        result = new LayerConvolution(json, input_geometry);

    if ( (type == "max pooling") || (type == "max_pooling"))
        result = new LayerMaxPooling(json, input_geometry);

    if (type == "output")
    {
        result = new LayerOutput(json, input_geometry);
 
        this->output_geometry.w = result->get_output_geometry().w;
        this->output_geometry.h = result->get_output_geometry().h;
        this->output_geometry.d = result->get_output_geometry().d;
    }


    return result;
}
