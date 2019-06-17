#include <layers/unpooling_layer.h>

#include <kernels/unpooling_layer.cuh>

UnPoolingLayer::UnPoolingLayer()
        :Layer()
{

}

UnPoolingLayer::UnPoolingLayer(UnPoolingLayer& other)
        :Layer(other)
{

}

UnPoolingLayer::UnPoolingLayer(const UnPoolingLayer& other)
        :Layer(other)
{

}

UnPoolingLayer::UnPoolingLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_average_pooling_layer();
}

UnPoolingLayer::~UnPoolingLayer()
{

}

UnPoolingLayer& UnPoolingLayer::operator= (UnPoolingLayer& other)
{
    copy(other);
    return *this;
}

UnPoolingLayer& UnPoolingLayer::operator= (const UnPoolingLayer& other)
{
    copy(other);
    return *this;
}


std::string UnPoolingLayer::asString()
{
    std::string result;

    result+= "UNPOOLING\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}


void UnPoolingLayer::forward(Tensor &output, Tensor &input)
{
    unpooling_layer_forward(output, input);
}

void UnPoolingLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights)
{
    (void)input;
    (void)output;
    (void)update_weights;

    unpooling_layer_backward(error_back, error);
}


void UnPoolingLayer::init_average_pooling_layer()
{
    unsigned int kw = m_parameters["shape"][0].asInt();
    unsigned int kh = m_parameters["shape"][1].asInt();

    this->m_output_shape.set(this->m_input_shape.w()*kw, this->m_input_shape.h()*kh, this->m_input_shape.d());
    this->m_trainable_parameters    = 0;
    this->m_flops                   = this->m_output_shape.size()*kw*kh*2;
}
