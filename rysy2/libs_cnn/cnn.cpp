#include <cnn.h>

#include <json_config.h>

#include <layers/activation_elu_layer.h>
#include <layers/activation_relu_layer.h>

#include <layers/convolution_layer.h>
#include <layers/dense_convolution_layer.h>
#include <layers/fc_layer.h>


#include <layers/max_pooling_layer.h>
#include <layers/average_pooling_layer.h>
#include <layers/dropout_layer.h>

#include <iostream>


CNN::CNN()
{
    this->m_hyperparameters = default_hyperparameters();

    this->training_mode       = false;
    this->minibatch_counter   = 0;

    this->m_total_flops = 0;
    this->m_total_trainable_parameters = 0;
}

CNN::CNN(CNN& other)
{
    copy(other);
}

CNN::CNN(const CNN& other)
{
    copy(other);
}

CNN::CNN(std::string json_file_name, Shape input_shape, Shape output_shape)
{
    JsonConfig json(json_file_name);
    init(json.result, input_shape, output_shape);
}

CNN::CNN(Json::Value json_config, Shape input_shape, Shape output_shape)
{
    init(json_config, input_shape, output_shape);
}

CNN::CNN(Shape input_shape, Shape output_shape, float learning_rate, float lambda1, float lambda2, float dropout, unsigned int minibatch_size)
{
    Json::Value parameters;

    parameters["hyperparameters"] = default_hyperparameters();
    parameters["hyperparameters"]["learning_rate"] = learning_rate;
    parameters["hyperparameters"]["lambda1"] = lambda1;
    parameters["hyperparameters"]["lambda2"] = lambda2;
    parameters["hyperparameters"]["dropout"] = dropout;
    parameters["hyperparameters"]["minibatch_size"] = minibatch_size;

    init(parameters, input_shape, output_shape);
}

CNN::~CNN()
{
    for (unsigned int i = 0; i < layers.size(); i++)
    {
        delete layers[i];
        layers[i] = nullptr;
    }
}

CNN& CNN::operator= (CNN& other)
{
    copy(other);
    return *this;
}

CNN& CNN::operator= (const CNN& other)
{
    copy(other);
    return *this;
}

void CNN::copy(CNN& other)
{
    this->m_input_shape     = other.m_input_shape;
    this->m_output_shape    = other.m_output_shape;

    this->m_hyperparameters = other.m_hyperparameters;

    this->output            = other.output;
    this->required_output   = other.required_output;
    this->input             = other.input;

    this->layers            = other.layers;

    this->l_error           = other.l_error;
    this->l_output          = other.l_output;

    this->training_mode     = other.training_mode;
    this->minibatch_counter = other.minibatch_counter;

    this->m_total_flops = other.m_total_flops;
    this->m_total_trainable_parameters = other.m_total_trainable_parameters;
}

void CNN::copy(const CNN& other)
{
    this->m_input_shape     = other.m_input_shape;
    this->m_output_shape    = other.m_output_shape;

    this->m_hyperparameters = other.m_hyperparameters;

    this->output            = other.output;
    this->required_output   = other.required_output;
    this->input             = other.input;

    this->layers            = other.layers;

    this->l_error           = other.l_error;
    this->l_output          = other.l_output;

    this->training_mode     = other.training_mode;
    this->minibatch_counter = other.minibatch_counter;

    this->m_total_flops = other.m_total_flops;
    this->m_total_trainable_parameters = other.m_total_trainable_parameters;
}



void CNN::forward(Tensor &output, Tensor &input)
{
    this->output.clear();

}

void CNN::forward(std::vector<float> &output, std::vector<float> &input)
{
    this->input.set_from_host(input);

    forward(this->output, this->input);

    this->output.set_to_host(output);
}

void CNN::forward(std::vector<std::vector<float>> &output, std::vector<std::vector<float>> &input)
{
    for (unsigned int i = 0; i < output.size(); i++)
        forward(output[i], input[i]);
}


void CNN::train(Tensor &required_output, Tensor &input)
{

}

void CNN::train(std::vector<float> &required_output, std::vector<float> &input)
{
    this->required_output.set_from_host(required_output);
    this->input.set_from_host(input);

    train(this->required_output, this->input);
}

void CNN::train(std::vector<Tensor> &required_output, std::vector<Tensor> &input)
{
    auto indices = make_indices(required_output.size());

    for (unsigned int i = 0; i < required_output.size(); i++)
        train(required_output[indices[i]], input[indices[i]]);
}


void CNN::set_training_mode()
{
    training_mode = true;

    for (unsigned int i = 0; i < layers.size(); i++)
        layers[i]->set_training_mode();
}

void CNN::unset_training_mode()
{
    training_mode = false;

    for (unsigned int i = 0; i < layers.size(); i++)
        layers[i]->unset_training_mode();
}

bool CNN::is_training_mode()
{
    return training_mode;
}

void CNN::reset()
{
    for (unsigned int i = 0; i < layers.size(); i++)
        layers[i]->reset();
}

void CNN::train(std::vector<std::vector<float>> &required_output, std::vector<std::vector<float>> &input)
{
    auto indices = make_indices(required_output.size());

    for (unsigned int i = 0; i < required_output.size(); i++)
        train(required_output[indices[i]], input[indices[i]]);
}

void CNN::init(Json::Value json_config, Shape input_shape, Shape output_shape)
{
    this->m_input_shape       = input_shape;
    this->m_output_shape      = output_shape;
    this->training_mode       = false;
    this->minibatch_counter   = 0;

    this->m_total_flops = 0;
    this->m_total_trainable_parameters = 0;

    if (json_config["network_log_file_name"] != Json::Value::null)
    {
        std::string network_log_file_name = json_config["network_log_file_name"].asString();
        if (network_log_file_name.size() > 0)
            network_log.set_output_file_name(network_log_file_name);
    }

    network_log << "network init start\n\n";

    if (json_config["hyperparameters"]["learning_rate"] != Json::Value::null)
        m_hyperparameters["learning_rate"]     = json_config["hyperparameters"]["learning_rate"].asFloat();
    else
        m_hyperparameters["learning_rate"] = default_hyperparameters()["learning_rate"];
;

    if (json_config["hyperparameters"]["lambda1"] != Json::Value::null)
        m_hyperparameters["lambda1"]            = json_config["hyperparameters"]["lambda1"].asFloat();
    else
        m_hyperparameters["lambda1"] = default_hyperparameters()["lambda1"];

    if (json_config["hyperparameters"]["lambda2"] != Json::Value::null)
        m_hyperparameters["lambda2"]            = json_config["hyperparameters"]["lambda2"].asFloat();
    else
        m_hyperparameters["lambda2"] = default_hyperparameters()["lambda2"];

    if (json_config["hyperparameters"]["dropout"] != Json::Value::null)
        m_hyperparameters["dropout"]            = json_config["hyperparameters"]["dropout"].asFloat();
    else
        m_hyperparameters["dropout"] = default_hyperparameters()["dropout"];

    if (json_config["hyperparameters"]["minibatch_size"] != Json::Value::null)
        m_hyperparameters["minibatch_size"]  = json_config["hyperparameters"]["minibatch_size"].asInt();
    else
        m_hyperparameters["minibatch_size"]  = default_hyperparameters()["minibatch_size"];

    network_log << "hyperparameters :\n";
    network_log << "learning_rate  = " << m_hyperparameters["learning_rate"].asFloat() << "\n";
    network_log << "lambda1        = " << m_hyperparameters["lambda1"].asFloat() << "\n";
    network_log << "lambda2        = " << m_hyperparameters["lambda2"].asFloat() << "\n";
    network_log << "dropout        = " << m_hyperparameters["dropout"].asFloat() << "\n";
    network_log << "minibatch_size = " << m_hyperparameters["minibatch_size"].asInt() << "\n";
    network_log << "\n\n";

    network_log << "input_shape  = " << m_input_shape.w() << " " << m_input_shape.h() << " " << m_input_shape.d() << "\n";
    network_log << "output_shape = " << m_output_shape.w() << " " << m_output_shape.h() << " " << m_output_shape.d() << "\n";
    network_log << "\n\n";

    l_error.push_back(Tensor(input_shape));
    l_output.push_back(Tensor(input_shape));

    m_current_input_shape = input_shape;
}


Shape CNN::add_layer(std::string layer_type, Shape shape)
{
    Shape output_shape;

    Json::Value parameters;

    parameters["hyperparameters"]   = m_hyperparameters;
    parameters["shape"][0]          = shape.w();
    parameters["shape"][1]          = shape.h();
    parameters["shape"][2]          = shape.d();

    Layer *layer = nullptr;

    if (layer_type == "elu")
    {
        layer = new ActivationEluLayer(m_current_input_shape, parameters);
    }
    else
    if (layer_type == "relu")
    {
        layer = new ActivationReluLayer(m_current_input_shape, parameters);
    }
    else
    if (layer_type == "convolution")
    {
        layer = new ConvolutionLayer(m_current_input_shape, parameters);
    }
    else
    if ((layer_type == "dense_convolution")||(layer_type == "dense convolution")||(layer_type == "dense_conv"))
    {
        layer = new DenseConvolutionLayer(m_current_input_shape, parameters);
    }
    else
    if (layer_type == "fc")
    {
        layer = new FCLayer(m_current_input_shape, parameters);
    }
    if (layer_type == "output")
    {
        parameters["shape"][0] = m_output_shape.w();
        parameters["shape"][1] = m_output_shape.h();
        parameters["shape"][2] = m_output_shape.d();
        layer = new FCLayer(m_current_input_shape, parameters);
    }
    else
    if ((layer_type == "max_pooling")||(layer_type == "max pooling"))
    {
        layer = new MaxPoolingLayer(m_current_input_shape, parameters);
    }
    else
    if ((layer_type == "average_pooling")||(layer_type == "average pooling"))
    {
        layer = new AveragePoolingLayer(m_current_input_shape, parameters);
    }
    else
    if (layer_type == "dropout")
    {
        layer = new DropoutLayer(m_current_input_shape, parameters);
    }

    layers.push_back(layer);

    unsigned int layer_idx = layers.size()-1;

    output_shape = layers[layer_idx]->get_output_shape();

    l_error.push_back(Tensor(output_shape));
    l_output.push_back(Tensor(output_shape));

    m_current_input_shape = output_shape;

    m_total_flops+= layers[layer_idx]->get_flops();
    m_total_trainable_parameters+= layers[layer_idx]->get_trainable_parameters();

    return output_shape;
}

std::string CNN::asString()
{
    std::string result;
    for (unsigned int i = 0; i < layers.size(); i++)
        result+= layers[i]->asString() + "\n";
 
    result+= "\n\n";
    result+= "GFLOPS = " + std::to_string(m_total_flops/1000000000.0) + "\n";
    result+= "FLOPS = " + std::to_string(m_total_flops) + "\n";
    result+= "TRAINABLE PARAMETERS = " + std::to_string(m_total_trainable_parameters) + "\n";

    return result;
}

std::vector<unsigned int> CNN::make_indices(unsigned int count)
{
    std::vector<unsigned int> result(count);
    for (unsigned int i = 0; i < result.size(); i++)
        result[i] = i;

    for (unsigned int i = 0; i < result.size(); i++)
    {
        unsigned int idx_a = i;
        unsigned int idx_b = rand()%result.size();

        unsigned int tmp;

        tmp = result[idx_a];
        result[idx_a] = result[idx_b];
        result[idx_b] = tmp;
    }

    return result;
}

Json::Value CNN::default_hyperparameters(float learning_rate)
{
    Json::Value result;

    result["learning_rate"]  = learning_rate;
    result["lambda1"]        = learning_rate*0.001;
    result["lambda2"]        = learning_rate*0.001;
    result["minibatch_size"] = 32;
    result["dropout"]        = 0.5;

    return result;
}
