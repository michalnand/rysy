#include <layers/spatial_attention_layer.h>

#include <kernels/spatial_attention.cuh>

#include <iostream>

SpatialAttentionLayer::SpatialAttentionLayer()
        :Layer()
{

}

SpatialAttentionLayer::SpatialAttentionLayer(SpatialAttentionLayer& other)
        :Layer(other)
{

}

SpatialAttentionLayer::SpatialAttentionLayer(const SpatialAttentionLayer& other)
        :Layer(other)
{

}

SpatialAttentionLayer::SpatialAttentionLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_spatial_attention_layer();
}

SpatialAttentionLayer::~SpatialAttentionLayer()
{

}

SpatialAttentionLayer& SpatialAttentionLayer::operator= (SpatialAttentionLayer& other)
{
    copy(other);
    return *this;
}

SpatialAttentionLayer& SpatialAttentionLayer::operator= (const SpatialAttentionLayer& other)
{
    copy(other);
    return *this;
}


std::string SpatialAttentionLayer::asString()
{
    std::string result;

    result+= "S ATTENTION\t\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}


void SpatialAttentionLayer::forward(Tensor &output, Tensor &input)
{
    spatial_attention_forward(output, input);
}

void SpatialAttentionLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias)
{
    (void)output;
    (void)update_weights;
    (void)update_bias;

    spatial_attention_backward(error_back, input, error);
}


void SpatialAttentionLayer::init_spatial_attention_layer()
{
    this->m_output_shape.set(m_input_shape.w(), m_input_shape.h(), m_input_shape.d()/2);
    this->m_trainable_parameters    = 0;
    this->m_flops                   = 10*this->m_input_shape.size();
}
