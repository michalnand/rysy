#include <layers/recurrent_layer.h>
#include <iostream>

#include <kernels/fc_layer.cuh>
#include <kernels/activation_tanh_layer.cuh>
#include <kernels/solver_adam.cuh>
#include <math.h>

RecurrentLayer::RecurrentLayer()
        :Layer()
{

}

RecurrentLayer::RecurrentLayer(RecurrentLayer& other)
        :Layer(other)
{
    copy_recurrent_layer(other);
}

RecurrentLayer::RecurrentLayer(const RecurrentLayer& other)
        :Layer(other)
{
    copy_recurrent_layer(other);
}

RecurrentLayer::RecurrentLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_recurrent_layer();
}

RecurrentLayer::~RecurrentLayer()
{

}

RecurrentLayer& RecurrentLayer::operator= (RecurrentLayer& other)
{
    copy(other);
    copy_recurrent_layer(other);

    return *this;
}

RecurrentLayer& RecurrentLayer::operator= (const RecurrentLayer& other)
{
    copy(other);
    copy_recurrent_layer(other);

    return *this;
}


void RecurrentLayer::copy_recurrent_layer(RecurrentLayer &other)
{
    this->h                     = other.h;
    this->h_error               = other.h_error;
    this->time_sequence_length  = other.time_sequence_length;
}

void RecurrentLayer::copy_recurrent_layer(const RecurrentLayer &other)
{
    this->h                     = other.h;
    this->h_error               = other.h_error;
    this->time_sequence_length  = other.time_sequence_length;
}


void RecurrentLayer::reset()
{
    for (unsigned int i = 0; i < h.size(); i++)
        h[i].clear();

    for (unsigned int i = 0; i < h_error.size(); i++)
        h_error[i].clear();
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
        fc_input.concatenate(h[time_step_idx], input);

        fc_layer_forward(fc_output, fc_input, w, bias);

        activation_tanh_layer_forward(h[time_step_idx+1], fc_output);

        output = h[time_step_idx+1];
    }
    #ifdef RYSY_DEBUG
    else
    {
        std::cout << "RecurrentLayer::forward : exceed time_sequence_length " << time_step_idx << " expected " << time_sequence_length << "\n";
    }
    #endif
}

void RecurrentLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias)
{
    (void)output;
    (void)input;
    (void)update_weights;

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


    fc_input.concatenate(h[time_step_idx], input);
    h_error[time_step_idx + 1].add(error);

    activation_tanh_layer_backward(activation_error_back, h[time_step_idx+1], h_error[time_step_idx + 1]);


    fc_layer_gradient(w_grad, fc_input, activation_error_back);

    if (update_bias) 
        fc_layer_update_bias(bias, activation_error_back, learning_rate);

    if (update_weights)
    {
        solver_adam(w, w_grad, m, v, learning_rate, lambda1, lambda2, gradient_clip);
        w_grad.clear();
    }


    fc_layer_backward(fc_error_back, fc_input, activation_error_back, w);

    fc_error_back.split(h_error[time_step_idx], error_back);
}


void RecurrentLayer::save(std::string file_name_prefix)
{
    w.save(file_name_prefix + "_weights.bin");
    bias.save(file_name_prefix + "_bias.bin");
}

void RecurrentLayer::load(std::string file_name_prefix)
{
    w.load(file_name_prefix + "_weights.bin");
    bias.load(file_name_prefix + "_bias.bin");
}

std::string RecurrentLayer::asString()
{
    std::string result;

    result+= "RECURRENT\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + " " + std::to_string(m_input_shape.t()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + " " + std::to_string(m_output_shape.t()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}

void RecurrentLayer::init_recurrent_layer()
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
    time_sequence_length = m_parameters["hyperparameters"]["time_sequence_length"].asInt();


    unsigned int inputs_count  = m_input_shape.w()*m_input_shape.h()*m_input_shape.d();
    unsigned int neurons_count = w_*h_*d_;

    m_input_shape.set(1, 1, inputs_count);
    m_output_shape.set(1, 1, neurons_count);

    fc_input.init(1, 1, inputs_count + neurons_count);
    fc_output.init(1, 1, neurons_count);

    activation_error_back.init(1, 1, neurons_count);
    fc_error_back.init(1, 1, inputs_count + neurons_count);


    h.resize(time_sequence_length + 1);
    for (unsigned int i = 0; i < h.size(); i++)
        h[i].init(1, 1, neurons_count, 1);

    h_error.resize(time_sequence_length + 1);
    for (unsigned int i = 0; i < h_error.size(); i++)
        h_error[i].init(1, 1, neurons_count, 1);

    w.init(inputs_count + neurons_count, m_output_shape.size(), 1);
    w.set_random(sqrt(2.0/(inputs_count + neurons_count)));

    w_grad.init(w.shape());
    m.init(w.shape());
    v.init(w.shape());

    bias.init(1, 1, neurons_count);
    bias.clear();

    this->m_trainable_parameters    = w.size() + bias.size();
    this->m_flops                   = (inputs_count +neurons_count)*m_output_shape.size()*4;
}
