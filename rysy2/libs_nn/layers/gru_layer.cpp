#include <layers/gru_layer.h>
#include <iostream>

#include <kernels/fc_layer.cuh>
#include <kernels/gru_gate.cuh>
#include <kernels/solver_adam.cuh>

#include <math.h>

GRULayer::GRULayer()
        :Layer()
{

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
    this->time_sequence_length  = other.time_sequence_length;
    this->h                     = other.h;
    this->h_error               = other.h_error;

    this->fc_input              = other.fc_input;
    this->fc_output_control     = other.fc_output_control;
    this->fc_output_update      = other.fc_output_update;

    this->learning_rate         = other.learning_rate;
    this->lambda1               = other.lambda1;
    this->lambda2               = other.lambda2;
    this->gradient_clip         = other.gradient_clip;

    this->w_control             = other.w_control;
    this->bias_control          = other.bias_control;
    this->w_grad_control        = other.w_grad_control;
    this->m_control             = other.m_control;
    this->v_control             = other.v_control;


    this->w_update             = other.w_update;
    this->bias_update          = other.bias_update;
    this->w_grad_update        = other.w_grad_update;
    this->m_update             = other.m_update;
    this->v_update             = other.v_update;

    this->control_error_back   = other.control_error_back;
    this->update_error_back    = other.update_error_back;

    this->gate_control_error_back   = other.gate_control_error_back;
    this->gate_h_error_back         = other.gate_h_error_back;
    this->gate_update_error_back    = other.gate_update_error_back;
}

void GRULayer::copy_gru_layer(const GRULayer &other)
{
    this->time_sequence_length  = other.time_sequence_length;
    this->h                     = other.h;
    this->h_error               = other.h_error;

    this->fc_input              = other.fc_input;
    this->fc_output_control     = other.fc_output_control;
    this->fc_output_update      = other.fc_output_update;

    this->learning_rate         = other.learning_rate;
    this->lambda1               = other.lambda1;
    this->lambda2               = other.lambda2;
    this->gradient_clip         = other.gradient_clip;

    this->w_control             = other.w_control;
    this->bias_control          = other.bias_control;
    this->w_grad_control        = other.w_grad_control;
    this->m_control             = other.m_control;
    this->v_control             = other.v_control;


    this->w_update             = other.w_update;
    this->bias_update          = other.bias_update;
    this->w_grad_update        = other.w_grad_update;
    this->m_update             = other.m_update;
    this->v_update             = other.v_update;

    this->control_error_back   = other.control_error_back;
    this->update_error_back    = other.update_error_back;

    this->gate_control_error_back   = other.gate_control_error_back;
    this->gate_h_error_back         = other.gate_h_error_back;
    this->gate_update_error_back    = other.gate_update_error_back;
}


void GRULayer::reset()
{
    for (unsigned int i = 0; i < h.size(); i++)
        h[i].clear();

    for (unsigned int i = 0; i < h_error.size(); i++)
        h_error[i].clear();
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
        fc_input.concatenate(h[time_step_idx], input);

        fc_layer_forward(fc_output_control[time_step_idx], fc_input, w_control, bias_control);
        fc_layer_forward(fc_output_update[time_step_idx], fc_input, w_update, bias_update);


        gru_gate_forward(   h[time_step_idx+1],
                            fc_output_control[time_step_idx],
                            h[time_step_idx],
                            fc_output_update[time_step_idx]);

        output = h[time_step_idx+1];
    }
    #ifdef RYSY_DEBUG
    else
    {
        std::cout << "GRULayer::forward : exceed time_sequence_length " << time_step_idx << " expected " << time_sequence_length << "\n";
    }
    #endif
}

void GRULayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights)
{
    (void)output;
    (void)input;
    (void)update_weights;

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

    fc_input.concatenate(h[time_step_idx], input);
    h_error[time_step_idx + 1].add(error);


    gru_gate_backward(  h[time_step_idx + 1],

                        fc_output_control[time_step_idx],
                        h[time_step_idx],
                        fc_output_update[time_step_idx],

                        h_error[time_step_idx + 1],

                        gate_control_error_back,
                        gate_h_error_back,
                        gate_update_error_back);


    fc_layer_gradient(w_grad_control, fc_input, gate_control_error_back);
    fc_layer_update_bias(bias_control, gate_control_error_back, learning_rate);

    fc_layer_gradient(w_grad_update, fc_input, gate_update_error_back);
    fc_layer_update_bias(bias_update, gate_update_error_back, learning_rate);


    if (update_weights)
    {
        solver_adam(w_control, w_grad_control, m_control, v_control, learning_rate, lambda1, lambda2, gradient_clip);
        w_grad_control.clear();

        solver_adam(w_update, w_grad_update, m_update, v_update, learning_rate, lambda1, lambda2, gradient_clip);
        w_grad_update.clear();
    }


    fc_layer_backward(control_error_back, fc_input, gate_control_error_back, w_control);
    fc_layer_backward(update_error_back, fc_input, gate_update_error_back, w_update);

    control_error_back.add(update_error_back);
    control_error_back.split(h_error[time_step_idx], error_back);
}


void GRULayer::save(std::string file_name_prefix)
{
    w_control.save(file_name_prefix + "_weights_control.bin");
    bias_control.save(file_name_prefix + "_bias_control.bin");
    w_update.save(file_name_prefix + "_weights_update.bin");
    bias_update.save(file_name_prefix + "_bias_update.bin");
}

void GRULayer::load(std::string file_name_prefix)
{
    w_control.load(file_name_prefix + "_weights_control.bin");
    bias_control.load(file_name_prefix + "_bias_control.bin");
    w_update.load(file_name_prefix + "_weights_update.bin");
    bias_update.load(file_name_prefix + "_bias_update.bin");
}

std::string GRULayer::asString()
{
    std::string result;

    result+= "GRU\t\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + " " + std::to_string(m_input_shape.t()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + " " + std::to_string(m_output_shape.t()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}

void GRULayer::init_gru_layer()
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
    gradient_clip   = 1.0; //m_parameters["hyperparameters"]["gradient_clip"].asFloat();
    time_sequence_length = m_parameters["hyperparameters"]["time_sequence_length"].asInt();


    unsigned int inputs_count  = m_input_shape.w()*m_input_shape.h()*m_input_shape.d();
    unsigned int neurons_count = w_*h_*d_;

    m_input_shape.set(1, 1, inputs_count);
    m_output_shape.set(1, 1, neurons_count);

    fc_input.init(1, 1, inputs_count + neurons_count);

    fc_output_control.resize(time_sequence_length);
    fc_output_update.resize(time_sequence_length);

    for (unsigned int i = 0; i < fc_output_control.size(); i++)
        fc_output_control[i].init(1, 1, neurons_count);

    for (unsigned int i = 0; i < fc_output_update.size(); i++)
        fc_output_update[i].init(1, 1, neurons_count);



    h.resize(time_sequence_length + 1);
    for (unsigned int i = 0; i < h.size(); i++)
        h[i].init(1, 1, neurons_count, 1);

    h_error.resize(time_sequence_length + 1);
    for (unsigned int i = 0; i < h_error.size(); i++)
        h_error[i].init(1, 1, neurons_count, 1);

    gate_control_error_back.init(1, 1, neurons_count);
    gate_update_error_back.init(1, 1, neurons_count);
    gate_h_error_back.init(1, 1, neurons_count);

    control_error_back.init(1, 1, inputs_count + neurons_count);
    update_error_back.init(1, 1, inputs_count + neurons_count);



    w_control.init(inputs_count + neurons_count, m_output_shape.size(), 1);
    w_control.set_random(sqrt(2.0/(inputs_count + neurons_count)));

    w_grad_control.init(w_control.shape());
    m_control.init(w_control.shape());
    v_control.init(w_control.shape());

    bias_control.init(1, 1, neurons_count);
    bias_control.set_const(-1.0);


    w_update.init(inputs_count + neurons_count, m_output_shape.size(), 1);
    w_update.set_random(sqrt(2.0/(inputs_count + neurons_count)));

    w_grad_update.init(w_update.shape());
    m_update.init(w_update.shape());
    v_update.init(w_update.shape());

    bias_update.init(1, 1, neurons_count);
    bias_update.clear();



    this->m_trainable_parameters    = 2*(w_control.size() + bias_control.size());
    this->m_flops                   = (inputs_count +neurons_count)*m_output_shape.size()*4;
}
