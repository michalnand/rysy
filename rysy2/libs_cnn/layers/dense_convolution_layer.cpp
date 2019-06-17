#include <layers/dense_convolution_layer.h>

#include <kernels/convolution_layer_forward.cuh>
#include <kernels/convolution_layer_backward.cuh>
#include <kernels/solver_adam.cuh>

#include <cuda_float_allocator.cuh>
#include <cuda_tensor.cuh>

DenseConvolutionLayer::DenseConvolutionLayer()
        :Layer()
{

}

DenseConvolutionLayer::DenseConvolutionLayer(DenseConvolutionLayer& other)
        :Layer(other)
{
    copy_dense_convolution(other);
}

DenseConvolutionLayer::DenseConvolutionLayer(const DenseConvolutionLayer& other)
        :Layer(other)
{
    copy_dense_convolution(other);
}

DenseConvolutionLayer::DenseConvolutionLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_dense_convolution();
}

DenseConvolutionLayer::~DenseConvolutionLayer()
{

}

DenseConvolutionLayer& DenseConvolutionLayer::operator= (DenseConvolutionLayer& other)
{
    copy(other);
    copy_dense_convolution(other);
    return *this;
}

DenseConvolutionLayer& DenseConvolutionLayer::operator= (const DenseConvolutionLayer& other)
{
    copy(other);
    copy_dense_convolution(other);
    return *this;
}


void DenseConvolutionLayer::copy_dense_convolution(DenseConvolutionLayer &other)
{
    this->learning_rate     = other.learning_rate;
    this->lambda1           = other.lambda1;
    this->lambda2           = other.lambda2;

    this->w                 = other.w;
    this->bias              = other.bias;

    this->m_kernel_shape      = other.m_kernel_shape;

    this->m_conv_output       = other.m_conv_output;
}

void DenseConvolutionLayer::copy_dense_convolution(const DenseConvolutionLayer &other)
{
    this->learning_rate     = other.learning_rate;
    this->lambda1           = other.lambda1;
    this->lambda2           = other.lambda2;

    this->w                 = other.w;
    this->bias              = other.bias;

    this->m_kernel_shape     = other.m_kernel_shape;

    this->m_conv_output       = other.m_conv_output;
}


void DenseConvolutionLayer::forward(Tensor &output, Tensor &input)
{
    convolution_layer_forward(m_conv_output, input, w, bias);

    output.concatenate(m_conv_output, input);
}

void DenseConvolutionLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights)
{
    (void)output;

    cuda_float_allocator.device_to_device(m_error_convolution.v, error.v, m_error_convolution.size());

    convolution_layer_gradient(w_grad, input, error);
    convolution_layer_update_bias(bias, error, learning_rate);

    if (update_weights)
    {
        solver_adam(w, w_grad, m, v, learning_rate, lambda1, lambda2);
        w_grad.clear();
    }

    convolution_layer_backward(error_back, input, error, w);
    cuda_tensor_add(error_back.v, error.v + m_error_convolution.size(), error_back.size());

     //error_back.add()
}

void DenseConvolutionLayer::save(std::string file_name_prefix)
{
    w.save(file_name_prefix + "_weights.bin");
    bias.save(file_name_prefix + "_bias.bin");
}

void DenseConvolutionLayer::load(std::string file_name_prefix)
{
    w.load(file_name_prefix + "_weights.bin");
    bias.load(file_name_prefix + "_bias.bin");
}

std::string DenseConvolutionLayer::asString()
{
    std::string result;

    result+= "DENSE CONV\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_kernel_shape.w()) + " " + std::to_string(m_kernel_shape.h()) + " " + std::to_string(m_kernel_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}

void DenseConvolutionLayer::init_dense_convolution()
{
    unsigned int kw = m_parameters["shape"][0].asInt();
    unsigned int kh = m_parameters["shape"][1].asInt();
    unsigned int kd = m_parameters["shape"][2].asInt();

    m_kernel_shape.set(kw, kh, kd);


    learning_rate   = m_parameters["hyperparameters"]["learning_rate"].asFloat();
    lambda1         = m_parameters["hyperparameters"]["lambda1"].asFloat();
    lambda2         = m_parameters["hyperparameters"]["lambda2"].asFloat();

    m_conv_output.init(m_input_shape.w(), m_input_shape.h(), kd);
    m_output_shape.set(m_input_shape.w(), m_input_shape.h(), m_input_shape.d() + kd);

    w.init(kw, kh, kd*m_input_shape.d());
    w.set_random_xavier();

    w_grad.init(w.shape());
    m.init(w.shape());
    v.init(w.shape());

    bias.init(1, 1, kd);
    bias.clear();

    this->m_trainable_parameters    = w.size() + bias.size();
    this->m_flops                   = m_input_shape.w()*m_input_shape.h()*m_input_shape.d()*kw*kh*kd;
}
