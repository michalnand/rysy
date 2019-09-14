#include <layers/spatial_attention_layer.h>

#include <kernels/convolution_layer_forward.cuh>
#include <kernels/convolution_layer_backward.cuh>
#include <kernels/solver_adam.cuh>
#include <kernels/spatial_attention.cuh>

#include <math.h>

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
    copy_spatial_attention(other);
    return *this;
}

SpatialAttentionLayer& SpatialAttentionLayer::operator= (const SpatialAttentionLayer& other)
{
    copy(other);
    copy_spatial_attention(other);
    return *this;
}

void SpatialAttentionLayer::copy_spatial_attention(SpatialAttentionLayer &other)
{
    this->learning_rate     = other.learning_rate;
    this->lambda1           = other.lambda1;
    this->lambda2           = other.lambda2;
    this->gradient_clip     = other.gradient_clip;

    this->w                 = other.w;
    this->bias              = other.bias;

    this->m_kernel_shape      = other.m_kernel_shape;

    this->input_attention           = input_attention;
    this->error_back_attention      = error_back_attention;
    this->error_back_conv           = error_back_conv;
}

void SpatialAttentionLayer::copy_spatial_attention(const SpatialAttentionLayer &other)
{
    this->learning_rate     = other.learning_rate;
    this->lambda1           = other.lambda1;
    this->lambda2           = other.lambda2;
    this->gradient_clip     = other.gradient_clip;

    this->w                 = other.w;
    this->bias              = other.bias;

    this->m_kernel_shape      = other.m_kernel_shape;

    this->input_attention           = input_attention;
    this->error_back_attention      = error_back_attention;
    this->error_back_conv           = error_back_conv;
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
    #ifdef RYSY_DEBUG

    if (output.shape() != m_output_shape)
    {
        std::cout << "SpatialAttentionLayer::forward : inconsistent output shape ";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "SpatialAttentionLayer::forward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    convolution_layer_forward(input_attention, input, w, bias);
    spatial_attention_forward(output, input, input_attention);
}

void SpatialAttentionLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias)
{
    #ifdef RYSY_DEBUG

    if (error_back.shape() != m_input_shape)
    {
        std::cout << "SpatialAttentionLayer::backward : inconsistent error_back shape\n";
        error_back.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (error.shape() != m_output_shape)
    {
        std::cout << "SpatialAttentionLayer::backward : inconsistent error shape\n";
        error.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "SpatialAttentionLayer::backward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (output.shape() != m_output_shape)
    {
        std::cout << "SpatialAttentionLayer::backward : inconsistent output shape\n";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    (void)output;


    spatial_attention_backward( error_back,
                                error_back_attention,
                                input,
                                input_attention,
                                error);


    convolution_layer_gradient(w_grad, input, error_back_attention);

    if (update_bias)
        convolution_layer_update_bias(bias, error_back_attention, learning_rate);

    if (update_weights)
    {
        solver_adam(w, w_grad, m, v, learning_rate, lambda1, lambda2, gradient_clip);
        w_grad.clear();
    }


    convolution_layer_backward(error_back_conv, error_back_attention, w);

    error_back.add(error_back_conv);
}

void SpatialAttentionLayer::save(std::string file_name_prefix)
{
    w.save(file_name_prefix + "_weights.bin");
    bias.save(file_name_prefix + "_bias.bin");
}

void SpatialAttentionLayer::load(std::string file_name_prefix)
{
    w.load(file_name_prefix + "_weights.bin");
    bias.load(file_name_prefix + "_bias.bin");
}


void SpatialAttentionLayer::init_spatial_attention_layer()
{
    unsigned int kw = m_parameters["shape"][0].asInt();
    unsigned int kh = m_parameters["shape"][1].asInt();
    unsigned int kd = m_input_shape.d();

    m_kernel_shape.set(kw, kh, kd);

    learning_rate   = m_parameters["hyperparameters"]["learning_rate"].asFloat();
    lambda1         = m_parameters["hyperparameters"]["lambda1"].asFloat();
    lambda2         = m_parameters["hyperparameters"]["lambda2"].asFloat();
    gradient_clip   = m_parameters["hyperparameters"]["gradient_clip"].asFloat();

    m_output_shape.set(m_input_shape.w(), m_input_shape.h(), kd);

    w.init(kw, kh, kd*m_input_shape.d());
    w.set_random(sqrt(2.0/w.size()));

    w_grad.init(w.shape());
    m.init(w.shape());
    v.init(w.shape());

    bias.init(1, 1, kd);
    bias.set_random(0.000001);

    input_attention.init(m_output_shape);
    error_back_attention.init(m_output_shape);
    error_back_conv.init(m_input_shape);

    this->m_trainable_parameters    = w.size() + bias.size();
    this->m_flops                   = 10*this->m_output_shape.size() + m_input_shape.w()*m_input_shape.h()*m_input_shape.d()*kw*kh*kd;
}
