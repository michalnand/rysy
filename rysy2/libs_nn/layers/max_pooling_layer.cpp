#include <layers/max_pooling_layer.h>

#include <kernels/max_pooling_layer.cuh>

#include <iostream>

MaxPoolingLayer::MaxPoolingLayer()
        :Layer()
{

}

MaxPoolingLayer::MaxPoolingLayer(MaxPoolingLayer& other)
        :Layer(other)
{

}

MaxPoolingLayer::MaxPoolingLayer(const MaxPoolingLayer& other)
        :Layer(other)
{

}

MaxPoolingLayer::MaxPoolingLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_max_pooling_layer();
}

MaxPoolingLayer::~MaxPoolingLayer()
{

}

MaxPoolingLayer& MaxPoolingLayer::operator= (MaxPoolingLayer& other)
{
    copy(other);
    return *this;
}

MaxPoolingLayer& MaxPoolingLayer::operator= (const MaxPoolingLayer& other)
{
    copy(other);
    return *this;
}


std::string MaxPoolingLayer::asString()
{
    std::string result;

    result+= "MAX POOLING\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}


void MaxPoolingLayer::forward(Tensor &output, Tensor &input)
{
    #ifdef RYSY_DEBUG

    if (output.shape() != m_output_shape)
    {
        std::cout << "MaxPoolingLayer::forward : inconsistent output shape ";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "MaxPoolingLayer::forward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    max_pooling_layer_forward(max_mask, output, input);
}

void MaxPoolingLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias)
{
    (void)input;
    (void)output;
    (void)update_weights;
    (void)update_bias;

    #ifdef RYSY_DEBUG

    if (error_back.shape() != m_input_shape)
    {
        std::cout << "MaxPoolingLayer::backward : inconsistent error_back shape\n";
        error_back.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (error.shape() != m_output_shape)
    {
        std::cout << "MaxPoolingLayer::backward : inconsistent error shape\n";
        error.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "MaxPoolingLayer::backward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (output.shape() != m_output_shape)
    {
        std::cout << "MaxPoolingLayer::backward : inconsistent output shape\n";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    max_pooling_layer_backward(error_back, error, max_mask);
}


void MaxPoolingLayer::init_max_pooling_layer()
{
    unsigned int kw = m_parameters["shape"][0].asInt();
    unsigned int kh = m_parameters["shape"][1].asInt();

    this->m_output_shape.set(this->m_input_shape.w()/kw, this->m_input_shape.h()/kh, this->m_input_shape.d());
    this->m_trainable_parameters    = 0;
    this->m_flops                   = this->m_output_shape.size()*kw*kh*2;

    this->max_mask.init(m_input_shape);
}
