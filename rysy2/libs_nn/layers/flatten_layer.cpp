#include <layers/flatten_layer.h>

#include <cuda_float_allocator.cuh>

FlattenLayer::FlattenLayer()
        :Layer()
{

}

FlattenLayer::FlattenLayer(FlattenLayer& other)
        :Layer(other)
{

}

FlattenLayer::FlattenLayer(const FlattenLayer& other)
        :Layer(other)
{

}

FlattenLayer::FlattenLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_flatten_layer();
}

FlattenLayer::~FlattenLayer()
{

}

FlattenLayer& FlattenLayer::operator= (FlattenLayer& other)
{
    copy(other);
    return *this;
}

FlattenLayer& FlattenLayer::operator= (const FlattenLayer& other)
{
    copy(other);
    return *this;
}


std::string FlattenLayer::asString()
{
    std::string result;

    result+= "FLATTEN\t\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + " " + std::to_string(m_input_shape.t()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + " " + std::to_string(m_output_shape.t()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}


void FlattenLayer::forward(Tensor &output, Tensor &input)
{
    cuda_float_allocator.device_to_device(output.v, input.v, input.size());
}

void FlattenLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias)
{
    (void)input;
    (void)update_weights;
    (void)output;
    (void)update_bias;

    cuda_float_allocator.device_to_device(error_back.v, error.v, error.size());
}


void FlattenLayer::init_flatten_layer()
{
    this->m_output_shape.set(1, 1, m_input_shape.size());
    this->m_trainable_parameters    = 0;
    this->m_flops                   = m_input_shape.size();
}
