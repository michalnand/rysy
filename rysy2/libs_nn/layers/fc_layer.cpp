#include <layers/fc_layer.h>

#include <kernels/fc_layer.cuh>
#include <kernels/solver_adam.cuh>

#include <iostream>
#include <math.h>

FCLayer::FCLayer()
        :Layer()
{

}

FCLayer::FCLayer(FCLayer& other)
        :Layer(other)
{
    copy_fc(other);
}

FCLayer::FCLayer(const FCLayer& other)
        :Layer(other)
{
    copy_fc(other);
}

FCLayer::FCLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_fc();
}

FCLayer::~FCLayer()
{

}

FCLayer& FCLayer::operator= (FCLayer& other)
{
    copy(other);
    copy_fc(other);
    return *this;
}

FCLayer& FCLayer::operator= (const FCLayer& other)
{
    copy(other);
    copy_fc(other);
    return *this;
}


void FCLayer::copy_fc(FCLayer &other)
{
    this->learning_rate     = other.learning_rate;
    this->lambda1           = other.lambda1;
    this->lambda2           = other.lambda2;
    this->gradient_clip     = other.gradient_clip;


    this->w                 = other.w;
    this->bias              = other.bias;

    this->w_grad            = other.w_grad;
    this->m                 = other.m;
    this->v                 = other.v;
}

void FCLayer::copy_fc(const FCLayer &other)
{
    this->learning_rate     = other.learning_rate;
    this->lambda1           = other.lambda1;
    this->lambda2           = other.lambda2;
    this->gradient_clip     = other.gradient_clip;


    this->w                 = other.w;
    this->bias              = other.bias;

    this->w_grad            = other.w_grad;
    this->m                 = other.m;
    this->v                 = other.v;
}


void FCLayer::forward(Tensor &output, Tensor &input)
{
    #ifdef RYSY_DEBUG

    if (output.shape() != m_output_shape)
    {
        std::cout << "FCLayer::forward : inconsistent output shape ";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "FCLayer::forward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    fc_layer_forward(output, input, w, bias);
}

void FCLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights)
{
    (void)output;

    #ifdef RYSY_DEBUG

    if (error_back.shape() != m_input_shape)
    {
        std::cout << "FCLayer::backward : inconsistent error_back shape\n";
        error_back.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (error.shape() != m_output_shape)
    {
        std::cout << "FCLayer::backward : inconsistent error shape\n";
        error.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "FCLayer::backward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (output.shape() != m_output_shape)
    {
        std::cout << "FCLayer::backward : inconsistent output shape\n";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    fc_layer_gradient(w_grad, input, error);
    fc_layer_update_bias(bias, error, learning_rate);

    if (update_weights)
    {
        solver_adam(w, w_grad, m, v, learning_rate, lambda1, lambda2, gradient_clip);
        w_grad.clear();
    }

    fc_layer_backward(error_back, input, error, w);
}

void FCLayer::save(std::string file_name_prefix)
{
    w.save(file_name_prefix + "_weights.bin");
    bias.save(file_name_prefix + "_bias.bin");
}

void FCLayer::load(std::string file_name_prefix)
{
    w.load(file_name_prefix + "_weights.bin");
    bias.load(file_name_prefix + "_bias.bin");
}

std::string FCLayer::asString()
{
    std::string result;

    result+= "FC\t\t\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}

void FCLayer::init_fc()
{
    unsigned int w_ = 1, h_ = 1, d_ = 1;

    if (m_parameters["shape"].size() >= 1)
        w_ = m_parameters["shape"][0].asInt();

    if (m_parameters["shape"].size() >= 2)
        h_ = m_parameters["shape"][1].asInt();

    if (m_parameters["shape"].size() >= 3)
        d_ = m_parameters["shape"][2].asInt();

    learning_rate   = m_parameters["hyperparameters"]["learning_rate"].asFloat();
    lambda1         = m_parameters["hyperparameters"]["lambda1"].asFloat();
    lambda2         = m_parameters["hyperparameters"]["lambda2"].asFloat();
    gradient_clip   = m_parameters["hyperparameters"]["gradient_clip"].asFloat();

    m_output_shape.set(1, 1, w_*h_*d_);

    w.init(m_input_shape.size(), m_output_shape.size(), 1);
    w.set_random(sqrt(2.0/m_input_shape.size()));

    w_grad.init(w.shape());
    m.init(w.shape());
    v.init(w.shape());

    bias.init(1, 1, m_output_shape.size());
    bias.clear();

    this->m_trainable_parameters    = w.size() + bias.size();
    this->m_flops                   = (m_input_shape.size() + 1)*m_output_shape.size()*2;
}
