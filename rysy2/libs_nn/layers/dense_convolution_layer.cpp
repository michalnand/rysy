#include <layers/dense_convolution_layer.h>

#include <kernels/convolution_layer_forward.cuh>
#include <kernels/convolution_layer_backward.cuh>
#include <kernels/solver_adam.cuh>

#include <cuda_float_allocator.cuh>
#include <cuda_tensor.cuh>

#include <iostream>
#include <math.h>

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
    this->gradient_clip     = other.gradient_clip;


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
    this->gradient_clip     = other.gradient_clip;


    this->w                 = other.w;
    this->bias              = other.bias;

    this->m_kernel_shape     = other.m_kernel_shape;

    this->m_conv_output       = other.m_conv_output;
}


void DenseConvolutionLayer::forward(Tensor &output, Tensor &input)
{
    #ifdef RYSY_DEBUG

    if (output.shape() != m_output_shape)
    {
        std::cout << "DenseConvolutionLayer::forward : inconsistent output shape ";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "DenseConvolutionLayer::forward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    convolution_layer_forward(m_conv_output, input, w, bias);

    output.concatenate(m_conv_output, input);
}

void DenseConvolutionLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias)
{
    (void)output;

    #ifdef RYSY_DEBUG

    if (error_back.shape() != m_input_shape)
    {
        std::cout << "DenseConvolutionLayer::backward : inconsistent error_back shape\n";
        error_back.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (error.shape() != m_output_shape)
    {
        std::cout << "DenseConvolutionLayer::backward : inconsistent error shape\n";
        error.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "DenseConvolutionLayer::backward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (output.shape() != m_output_shape)
    {
        std::cout << "DenseConvolutionLayer::backward : inconsistent output shape\n";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    error.split(m_error_convolution, m_error_direct);


    convolution_layer_gradient(w_grad, input, m_error_convolution);

    if (update_bias)
        convolution_layer_update_bias(bias, m_error_convolution, learning_rate);

    if (update_weights)
    {
        solver_adam(w, w_grad, m, v, learning_rate, lambda1, lambda2, gradient_clip);
        w_grad.clear();
    }

    convolution_layer_backward(error_back, m_error_convolution, w);

    error_back.add(m_error_direct);
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
    gradient_clip   = m_parameters["hyperparameters"]["gradient_clip"].asFloat();


    m_conv_output.init(m_input_shape.w(), m_input_shape.h(), kd);
    m_output_shape.set(m_input_shape.w(), m_input_shape.h(), m_input_shape.d() + kd);

    m_error_convolution.init(m_input_shape.w(), m_input_shape.h(), kd);
    m_error_direct.init(m_input_shape.w(), m_input_shape.h(), m_input_shape.d());

    w.init(kw, kh, kd*m_input_shape.d());
    w.set_random(sqrt(2.0/m_input_shape.size()));

    w_grad.init(w.shape());
    m.init(w.shape());
    v.init(w.shape());

    bias.init(1, 1, kd);
    bias.clear();

    this->m_trainable_parameters    = w.size() + bias.size();
    this->m_flops                   = m_input_shape.w()*m_input_shape.h()*m_input_shape.d()*kw*kh*kd;
}
