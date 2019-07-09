#include <layers/recurrent_layer.h>

#include <kernels/fc_layer.cuh>
#include <kernels/rl_tanh_block_layer.cuh>

#include <kernels/solver_adam.cuh>

#include <iostream>
#include <math.h>

RecurrentLayer::RecurrentLayer()
        :Layer()
{
    time_step_idx = 0;
}

RecurrentLayer::RecurrentLayer(RecurrentLayer& other)
        :Layer(other)
{
    copy_rl(other);
}

RecurrentLayer::RecurrentLayer(const RecurrentLayer& other)
        :Layer(other)
{
    copy_rl(other);
}

RecurrentLayer::RecurrentLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_rl();
}

RecurrentLayer::~RecurrentLayer()
{

}

RecurrentLayer& RecurrentLayer::operator= (RecurrentLayer& other)
{
    copy(other);
    copy_rl(other);
    return *this;
}

RecurrentLayer& RecurrentLayer::operator= (const RecurrentLayer& other)
{
    copy(other);
    copy_rl(other);
    return *this;
}


void RecurrentLayer::copy_rl(RecurrentLayer &other)
{
    this->learning_rate     = other.learning_rate;
    this->lambda1           = other.lambda1;
    this->lambda2           = other.lambda2;
    this->gradient_clip     = other.gradient_clip;

    this->wx                = other.wx;
    this->wh                = other.wh;
    this->bias              = other.bias;

    this->wx_grad            = other.wx_grad;
    this->mx                 = other.mx;
    this->vx                 = other.vx;

    this->wh_grad            = other.wh_grad;
    this->mh                 = other.mh;
    this->vh                 = other.vh;

    this->h                  = other.h;
    this->error_h            = other.error_h;

    this->time_sequence_length  = other.time_sequence_length;
    this->time_step_idx      = other.time_step_idx;
}

void RecurrentLayer::copy_rl(const RecurrentLayer &other)
{
    this->learning_rate     = other.learning_rate;
    this->lambda1           = other.lambda1;
    this->lambda2           = other.lambda2;
    this->gradient_clip     = other.gradient_clip;

    this->wx                = other.wx;
    this->wh                = other.wh;
    this->bias              = other.bias;

    this->wx_grad            = other.wx_grad;
    this->mx                 = other.mx;
    this->vx                 = other.vx;

    this->wh_grad            = other.wh_grad;
    this->mh                 = other.mh;
    this->vh                 = other.vh;

    this->h                  = other.h;
    this->error_h            = other.error_h;

    this->time_sequence_length  = other.time_sequence_length;
    this->time_step_idx      = other.time_step_idx;
}


void RecurrentLayer::reset()
{
    for (unsigned int i = 0; i < h.size(); i++)
        h[i].clear();

    for (unsigned int i = 0; i < error_h.size(); i++)
        error_h[i].clear();

    time_step_idx = 0;
}


void RecurrentLayer::forward(Tensor &output, Tensor &input)
{
    #ifdef RYSY_DEBUG

    if (output.shape() != m_output_shape)
    {
        std::cout << "RecurrentLayer::forward : inconsistent output shape ";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "RecurrentLayer::forward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    if (time_step_idx < time_sequence_length)
    {
        rl_tanh_block_layer_forward(output, input, h[time_step_idx], wx, wh, bias);
        h[time_step_idx + 1] = output;
        time_step_idx++;
    }
    #ifdef RYSY_DEBUG
    else
    {
        std::cout << "RecurrentLayer::forward : exceed time_sequence_length " << time_step_idx << " expected " << time_sequence_length << "\n";
    }
    #endif
}

void RecurrentLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights)
{
    #ifdef RYSY_DEBUG

    if (error_back.shape() != m_input_shape)
    {
        std::cout << "RecurrentLayer::backward : inconsistent error_back shape\n";
        error_back.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (error.shape() != m_output_shape)
    {
        std::cout << "RecurrentLayer::backward : inconsistent error shape\n";
        error.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "RecurrentLayer::backward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (output.shape() != m_output_shape)
    {
        std::cout << "RecurrentLayer::backward : inconsistent output shape\n";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    rl_tanh_block_layer_gradient(wx_grad, wh_grad, input, h[time_step_idx], output, error);

    fc_layer_update_bias(bias, error, learning_rate);

    if (update_weights)
    {
        solver_adam(wx, wx_grad, mx, vx, learning_rate, lambda1, lambda2, gradient_clip);
        wx_grad.clear();

        solver_adam(wh, wh_grad, mh, vh, learning_rate, lambda1, lambda2, gradient_clip);
        wh_grad.clear();
    }

    rl_tanh_block_layer_backward(   error_back, error_h[time_step_idx-1],
                                    input, h[time_step_idx],
                                    output, error,
                                    wx, wh);
}

void RecurrentLayer::save(std::string file_name_prefix)
{
    wx.save(file_name_prefix + "_weights_x.bin");
    wh.save(file_name_prefix + "_weights_h.bin");

    bias.save(file_name_prefix + "_bias.bin");
}

void RecurrentLayer::load(std::string file_name_prefix)
{
    wx.load(file_name_prefix + "_weights_x.bin");
    wh.load(file_name_prefix + "_weights_h.bin");

    bias.load(file_name_prefix + "_bias.bin");
}

std::string RecurrentLayer::asString()
{
    std::string result;

    result+= "RECURRENT\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}

void RecurrentLayer::init_rl()
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


    time_step_idx   = 0;
    time_sequence_length = m_parameters["hyperparameters"]["time_sequence_length"].asInt();

    m_output_shape.set(1, 1, w_*h_*d_);


    wx.init(m_input_shape.size(), m_output_shape.size(), 1);
    wx.set_random(sqrt(2.0/m_input_shape.size()));

    wx_grad.init(wx.shape());
    mx.init(wx.shape());
    vx.init(wx.shape());


    Shape inputh_shape(m_output_shape.w(), m_output_shape.h(), m_output_shape.d());

    wh.init(inputh_shape.size(), m_output_shape.size(), 1);
    wh.set_random(sqrt(2.0/inputh_shape.size()));

    wh_grad.init(wh.shape());
    mh.init(wh.shape());
    vh.init(wh.shape());


    bias.init(1, 1, m_output_shape.size());
    bias.clear();

    h.resize(time_sequence_length+1);
    error_h.resize(time_sequence_length+1);

    for (unsigned int i = 0; i < h.size(); i++)
        h[i].init(inputh_shape.w(), inputh_shape.h(), inputh_shape.d());

    for (unsigned int i = 0; i < error_h.size(); i++)
        error_h[i].init(inputh_shape.w(), inputh_shape.h(), inputh_shape.d());



    this->m_trainable_parameters    = wx.size() + wh.size() + bias.size();
    this->m_flops                   = (m_input_shape.size() + inputh_shape.size() + 1)*m_output_shape.size()*2;
}
