#include <layers/average_pooling_layer.h>

#include <kernels/average_pooling_layer.cuh>

AveragePoolingLayer::AveragePoolingLayer()
        :Layer()
{

}

AveragePoolingLayer::AveragePoolingLayer(AveragePoolingLayer& other)
        :Layer(other)
{

}

AveragePoolingLayer::AveragePoolingLayer(const AveragePoolingLayer& other)
        :Layer(other)
{

}

AveragePoolingLayer::AveragePoolingLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_average_pooling_layer();
}

AveragePoolingLayer::~AveragePoolingLayer()
{

}

AveragePoolingLayer& AveragePoolingLayer::operator= (AveragePoolingLayer& other)
{
    copy(other);
    return *this;
}

AveragePoolingLayer& AveragePoolingLayer::operator= (const AveragePoolingLayer& other)
{
    copy(other);
    return *this;
}


std::string AveragePoolingLayer::asString()
{
    std::string result;

    result+= "AVERAGE POOLING\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}


void AveragePoolingLayer::forward(Tensor &output, Tensor &input)
{
    average_pooling_layer_forward(output, input);
}

void AveragePoolingLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights)
{
    (void)input;
    (void)output;
    (void)update_weights;

    average_pooling_layer_backward(error_back, error);
}


void AveragePoolingLayer::init_average_pooling_layer()
{
    unsigned int kw = m_parameters["shape"][0].asInt();
    unsigned int kh = m_parameters["shape"][1].asInt();

    this->m_output_shape.set(this->m_input_shape.w()/kw, this->m_input_shape.h()/kh, this->m_input_shape.d());
    this->m_trainable_parameters    = 0;
    this->m_flops                   = this->m_output_shape.size()*kw*kh*2;
}
