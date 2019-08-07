#include <layers/gru_layer.h>
#include <math.h>

#include <kernels/fc_layer.cuh>
#include <kernels/gru_gate.cuh>

#include <iostream>

GRULayer::GRULayer()
        :Layer()
{
    time_step_idx = 0;
}

GRULayer::GRULayer(GRULayer& other)
        :Layer(other)
{
    copy_gru_layer(other);
}

GRULayer::GRULayer(const GRULayer& other)
        :Layer(other)
{
    copy_gru_layer(other);
}

GRULayer::GRULayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_gru_layer();
}

GRULayer::~GRULayer()
{

}

GRULayer& GRULayer::operator= (GRULayer& other)
{
    copy(other);
    copy_gru_layer(other);

    return *this;
}

GRULayer& GRULayer::operator= (const GRULayer& other)
{
    copy(other);
    copy_gru_layer(other);

    return *this;
}


void GRULayer::copy_gru_layer(GRULayer &other)
{
    this->h                     = other.h;
    this->time_sequence_length  = other.time_sequence_length;
    this->time_step_idx         = other.time_step_idx;
}

void GRULayer::copy_gru_layer(const GRULayer &other)
{
    this->h                     = other.h;
    this->time_sequence_length  = other.time_sequence_length;
    this->time_step_idx         = other.time_step_idx;
}


void GRULayer::reset()
{
    for (unsigned int i = 0; i < h.size(); i++)
        h[i].clear();

    for (unsigned int i = 0; i < control_output.size(); i++)
        control_output[i].clear();

    for (unsigned int i = 0; i < update_output.size(); i++)
        update_output[i].clear();

    for (unsigned int i = 0; i < h_error.size(); i++)
        h_error[i].clear();


    block_input.clear();
    control_h_error_back.clear();
    update_h_error_back.clear();

    control_error_back.clear();
    update_error_back.clear();
    tmp_error.clear();
    tmp_error_h.clear();

    time_step_idx = 0;
}


void GRULayer::forward(Tensor &output, Tensor &input)
{
    #ifdef RYSY_DEBUG

    if (output.shape() != m_output_shape)
    {
        std::cout << "GRULayer::forward : inconsistent output shape ";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "GRULayer::forward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    #endif


    if (time_step_idx < time_sequence_length)
    {
        block_input.concatenate(h[time_step_idx], input);

        fc_layer_forward(control_output[time_step_idx], block_input, control_weights.weights, control_bias);
        fc_layer_forward(update_output[time_step_idx], block_input, update_weights.weights, update_bias);

        gru_gate_forward(h[time_step_idx+1], control_output[time_step_idx], h[time_step_idx], update_output[time_step_idx]);

        output = h[time_step_idx+1];

        time_step_idx++;
    }
    #ifdef RYSY_DEBUG
    else
    {
        std::cout << "GRULayer::forward : exceed time_sequence_length " << time_step_idx << " expected " << time_sequence_length << "\n";
    }
    #endif
}



void GRULayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights_)
{
    #ifdef RYSY_DEBUG

    if (error_back.shape() != m_input_shape)
    {
        std::cout << "GRULayer::backward : inconsistent error_back shape\n";
        error_back.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (error.shape() != m_output_shape)
    {
        std::cout << "GRULayer::backward : inconsistent error shape\n";
        error.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "GRULayer::backward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (output.shape() != m_output_shape)
    {
        std::cout << "GRULayer::backward : inconsistent output shape\n";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    time_step_idx--;

    h_error[time_step_idx+1].add(error);

    gru_gate_backward(  h[time_step_idx+1],

                        control_output[time_step_idx],
                        h[time_step_idx],
                        update_output[time_step_idx],

                        h_error[time_step_idx+1],

                        control_h_error_back,
                        h_error[time_step_idx],
                        update_h_error_back);

    block_input.concatenate(h[time_step_idx], input);

    fc_layer_gradient(control_weights.gradient, block_input, control_h_error_back);
    fc_layer_gradient(update_weights.gradient, block_input, update_h_error_back);

    fc_layer_update_bias(control_bias, update_error_back, learning_rate);
    fc_layer_update_bias(update_bias, control_error_back, learning_rate);


    if (update_weights_)
    {
        control_weights.train(learning_rate, lambda1, lambda2, gradient_clip);
        update_weights.train(learning_rate, lambda1, lambda2, gradient_clip);
    }


    fc_layer_backward(control_error_back, block_input, control_h_error_back, control_weights.weights);
    fc_layer_backward(update_error_back, block_input, control_h_error_back, control_weights.weights);

    tmp_error = control_error_back;
    tmp_error.add(update_error_back);

    tmp_error.split(tmp_error_h, error_back);

    h_error[time_step_idx].add(tmp_error_h);

}



std::string GRULayer::asString()
{
    std::string result;

    result+= "GRU LAYER\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + " " + std::to_string(m_output_shape.t()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}

void GRULayer::init_gru_layer()
{
    time_step_idx        = 0;
    time_sequence_length = m_parameters["hyperparameters"]["time_sequence_length"].asInt();

    learning_rate   = m_parameters["hyperparameters"]["learning_rate"].asFloat();
    lambda1         = m_parameters["hyperparameters"]["lambda1"].asFloat();
    lambda2         = m_parameters["hyperparameters"]["lambda2"].asFloat();
    gradient_clip   = m_parameters["hyperparameters"]["gradient_clip"].asFloat();

    unsigned int w_ = 1, h_ = 1, d_ = 1;

    if (m_parameters["shape"].size() >= 1)
        w_ = m_parameters["shape"][0].asInt();

    if (m_parameters["shape"].size() >= 2)
        h_ = m_parameters["shape"][1].asInt();

    if (m_parameters["shape"].size() >= 3)
        d_ = m_parameters["shape"][2].asInt();

    unsigned int input_size = m_input_shape.size();
    unsigned int output_size = w_*h_*d_;

    m_output_shape.set(1, 1, output_size);

    control_weights.init(m_input_shape.size(), m_output_shape.size(), 1);
    update_weights.init(m_input_shape.size(), m_output_shape.size(), 1);

    control_bias.init(m_output_shape.size());
    update_bias.init(m_output_shape.size());

    control_weights.weights.set_random(sqrt(2.0/input_size));
    update_weights.weights.set_random(sqrt(2.0/input_size));
    control_bias.set_const(-1.0);
    update_bias.set_const(0.0);


    //init tensors

    h.resize(time_sequence_length+1);
    control_output.resize(time_sequence_length+1);
    update_output.resize(time_sequence_length+1);
    h_error.resize(time_sequence_length+1);

    for (unsigned int i = 0; i < h.size(); i++)
        h[i].init(m_output_shape);

    for (unsigned int i = 0; i < control_output.size(); i++)
        control_output[i].init(m_output_shape);

    for (unsigned int i = 0; i < update_output.size(); i++)
        update_output[i].init(m_output_shape);

    for (unsigned int i = 0; i < h_error.size(); i++)
        h_error[i].init(m_output_shape);

    block_input.init(m_input_shape.w(), m_input_shape.h(), m_input_shape.d() + m_output_shape.d());

    control_h_error_back.init(m_output_shape);
    update_h_error_back.init(m_output_shape);

    control_error_back.init(m_input_shape.w(), m_input_shape.h(), m_input_shape.d() + m_output_shape.d());
    update_error_back.init(m_input_shape.w(), m_input_shape.h(), m_input_shape.d() + m_output_shape.d());
    tmp_error.init(m_input_shape.w(), m_input_shape.h(), m_input_shape.d() + m_output_shape.d());
    tmp_error_h.init(m_output_shape);



    this->m_trainable_parameters    = control_weights.weights.size() + control_bias.size() + update_weights.weights.size() + update_bias.size();
    this->m_flops                   = m_output_shape.size()*( 2*(block_input.size() + 1) + 4 + 10);
}
