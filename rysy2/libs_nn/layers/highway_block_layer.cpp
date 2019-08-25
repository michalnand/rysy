#include <layers/highway_block_layer.h>

#include <kernels/highway_block.cuh>

#include <iostream>
#include <math.h>

HighwayBlockLayer::HighwayBlockLayer()
        :Layer()
{

}

HighwayBlockLayer::HighwayBlockLayer(HighwayBlockLayer& other)
        :Layer(other)
{

}

HighwayBlockLayer::HighwayBlockLayer(const HighwayBlockLayer& other)
        :Layer(other)
{

}

HighwayBlockLayer::HighwayBlockLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_highway_block_layer();
}

HighwayBlockLayer::~HighwayBlockLayer()
{

}

HighwayBlockLayer& HighwayBlockLayer::operator= (HighwayBlockLayer& other)
{
    copy(other);
    return *this;
}

HighwayBlockLayer& HighwayBlockLayer::operator= (const HighwayBlockLayer& other)
{
    copy(other);
    return *this;
}



void HighwayBlockLayer::forward(Tensor &output, Tensor &input)
{
    #ifdef RYSY_DEBUG

    if (output.shape() != m_output_shape)
    {
        std::cout << "HighwayBlockLayer::forward : inconsistent output shape ";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "HighwayBlockLayer::forward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    highway_layer_forward(output, input);
}

void HighwayBlockLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias)
{
    #ifdef RYSY_DEBUG

    if (error_back.shape() != m_input_shape)
    {
        std::cout << "HighwayBlockLayer::backward : inconsistent error_back shape\n";
        error_back.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (error.shape() != m_output_shape)
    {
        std::cout << "HighwayBlockLayer::backward : inconsistent error shape\n";
        error.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "HighwayBlockLayer::backward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (output.shape() != m_output_shape)
    {
        std::cout << "HighwayBlockLayer::backward : inconsistent output shape\n";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    (void)output;
    (void)update_weights;
    (void)update_bias;

    highway_layer_backward(error_back,input, error);
}


std::string HighwayBlockLayer::asString()
{
    std::string result;

    result+= "HIGHWAY\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}

void HighwayBlockLayer::init_highway_block_layer()
{
    #ifdef RYSY_DEBUG
        if (m_input_shape.d()%3 != 0)
        {
            std::cout << "HighwayBlockLayer::init_highway_block_layer : input d must by divisible by 3 : " << m_input_shape.d() << " but given\n";
            return;
        }
    #endif
    m_output_shape.set(m_input_shape.w(), m_input_shape.h(), m_input_shape.d()/3);

    unsigned int flops_tmp = 1 + 1 + 1 + 1 + 4*2;
    this->m_flops          = m_output_shape.size()*flops_tmp;
}
