#include <layers/max_pooling_layer.h>

#include <kernels/max_pooling_layer.cuh>

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
    max_pooling_layer_forward(max_mask, output, input);
}

void MaxPoolingLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights)
{
    (void)input;
    (void)output;
    (void)update_weights;

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
