#include <layers/fc_layer.h>
#include <kernels/fc_layer.cuh>
#include <kernels/solver_adam.cuh>

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

FCLayer::FCLayer(Shape input_shape, Json::Value parameters, unsigned int max_time_steps)
        :Layer(input_shape, parameters, max_time_steps)
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
    this->m_hyperparameters = other.m_hyperparameters;
    this->w                 = other.w;
    this->bias              = other.bias;
}

void FCLayer::copy_fc(const FCLayer &other)
{
    this->m_hyperparameters = other.m_hyperparameters;
    this->w                 = other.w;
    this->bias              = other.bias;
}


void FCLayer::forward(Tensor &output, Tensor &input)
{
    fc_layer_forward(output, input, w, bias);
}

void FCLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights)
{
    (void)input;
    
    fc_layer_gradient(w_grad, output, error);
    fc_layer_update_bias(bias, error, learning_rate);

     if (update_weights)
     {
         solver_adam(w, w_grad, m, v, learning_rate);
         w_grad.clear();
     }

     fc_layer_backward(error_back, output, error, w);
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

void FCLayer::init_fc()
{
    unsigned int w_ = 1, h_ = 1, d_ = 1;

    if (m_parameters["shape"].size() >= 1)
        w_ = m_parameters["shape"][0].asInt();

    if (m_parameters["shape"].size() >= 2)
        h_ = m_parameters["shape"][1].asInt();

    if (m_parameters["shape"].size() >= 3)
        d_ = m_parameters["shape"][2].asInt();

    m_hyperparameters = m_parameters["hyperparameters"];


    m_output_shape.set(1, 1, w_*h_*d_);

    w.init(m_input_shape.size(), m_output_shape.size(), 1);
    w.set_random_xavier();

    w_grad.init(w.shape());
    m.init(w.shape());
    v.init(w.shape());

    bias.init(1, 1, m_output_shape.size());
    bias.clear();
}
