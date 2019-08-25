#include <layers/average_pooling_layer.h>

#include <kernels/average_pooling_layer.cuh>

#include <iostream>

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
    #ifdef RYSY_DEBUG

    if (output.shape() != m_output_shape)
    {
        std::cout << "AveragePoolingLayer::forward : inconsistent output shape ";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "AveragePoolingLayer::forward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    average_pooling_layer_forward(output, input);
}

void AveragePoolingLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias)
{
    (void)input;
    (void)output;
    (void)update_weights;
    (void)update_bias;


    #ifdef RYSY_DEBUG

    if (error_back.shape() != m_input_shape)
    {
        std::cout << "AveragePoolingLayer::backward : inconsistent error_back shape\n";
        error_back.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (error.shape() != m_output_shape)
    {
        std::cout << "AveragePoolingLayer::backward : inconsistent error shape\n";
        error.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "AveragePoolingLayer::backward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (output.shape() != m_output_shape)
    {
        std::cout << "AveragePoolingLayer::backward : inconsistent output shape\n";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    #endif


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
