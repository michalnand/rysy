#include <layers/crop_layer.h>

#include <kernels/crop_layer.cuh>

CropLayer::CropLayer()
        :Layer()
{

}

CropLayer::CropLayer(CropLayer& other)
        :Layer(other)
{

}

CropLayer::CropLayer(const CropLayer& other)
        :Layer(other)
{

}

CropLayer::CropLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_crop_layer();
}

CropLayer::~CropLayer()
{

}

CropLayer& CropLayer::operator= (CropLayer& other)
{
    copy(other);
    return *this;
}

CropLayer& CropLayer::operator= (const CropLayer& other)
{
    copy(other);
    return *this;
}


std::string CropLayer::asString()
{
    std::string result;

    result+= "CROP\t\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}


void CropLayer::forward(Tensor &output, Tensor &input)
{
    crop_layer_forward(output, input);
}

void CropLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights)
{
    (void)input;
    (void)update_weights;
    (void)output;

    crop_layer_backward(error_back, error);
}


void CropLayer::init_crop_layer()
{
    this->m_output_shape.set(m_input_shape.w() - 2, m_input_shape.h() - 2, m_input_shape.d());
    this->m_trainable_parameters    = 0;
    this->m_flops                   = 2*this->m_input_shape.size();
}
