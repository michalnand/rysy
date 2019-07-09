#include <rnn.h>

#include <iostream>
#include <json_config.h>

#include <layers/activation_elu_layer.h>
#include <layers/activation_relu_layer.h>

#include <layers/convolution_layer.h>
#include <layers/dense_convolution_layer.h>
#include <layers/fc_layer.h>

#include <layers/recurrent_layer.h>

#include <layers/max_pooling_layer.h>
#include <layers/average_pooling_layer.h>
#include <layers/unpooling_layer.h>

#include <layers/dropout_layer.h>
#include <layers/crop_layer.h>

#include <svg_visualiser.h>

RNN::RNN()
{
    this->m_hyperparameters = default_hyperparameters();


    this->training_mode       = false;
    this->minibatch_counter   = 0;
    this->minibatch_size      = 32;

    this->time_sequence_length = 1;
    this->time_step_idx = 0;

    this->m_total_flops = 0;
    this->m_total_trainable_parameters = 0;
}

RNN::RNN(RNN& other)
{
    copy(other);
}

RNN::RNN(const RNN& other)
{
    copy(other);
}

RNN::RNN(std::string network_config_file_name, Shape input_shape, Shape output_shape)
{
    JsonConfig json(network_config_file_name);
    init(json.result, input_shape, output_shape);
}

RNN::RNN(Json::Value json_config, Shape input_shape, Shape output_shape)
{
    init(json_config, input_shape, output_shape);
}

RNN::RNN(Shape input_shape, Shape output_shape, float learning_rate, float lambda1, float lambda2, float gradient_clip, float dropout, unsigned int minibatch_size, unsigned int time_sequence_length)
{
    Json::Value parameters;

    parameters["hyperparameters"] = default_hyperparameters();
    parameters["hyperparameters"]["learning_rate"] = learning_rate;
    parameters["hyperparameters"]["lambda1"] = lambda1;
    parameters["hyperparameters"]["lambda2"] = lambda2;
    parameters["hyperparameters"]["gradient_clip"] = gradient_clip;
    parameters["hyperparameters"]["dropout"] = dropout;
    parameters["hyperparameters"]["minibatch_size"] = minibatch_size;

    init(parameters, input_shape, output_shape);
}

RNN::~RNN()
{
    for (unsigned int i = 0; i < layers.size(); i++)
    {
        delete layers[i];
        layers[i] = nullptr;
    }
}

RNN& RNN::operator= (RNN& other)
{
    copy(other);
    return *this;
}

RNN& RNN::operator= (const RNN& other)
{
    copy(other);
    return *this;
}

void RNN::copy(RNN& other)
{
    this->m_input_shape     = other.m_input_shape;
    this->m_output_shape    = other.m_output_shape;

    this->m_hyperparameters = other.m_hyperparameters;
    this->m_parameters      = other.m_parameters;

    this->output            = other.output;
    this->required_output   = other.required_output;
    this->input             = other.input;

    this->layers            = other.layers;

    this->l_error           = other.l_error;
    this->l_output          = other.l_output;

    this->training_mode     = other.training_mode;
    this->minibatch_counter = other.minibatch_counter;
    this->minibatch_size    = other.minibatch_size;

    this->time_sequence_length  = other.time_sequence_length;
    this->time_step_idx         = other.time_step_idx;

    this->m_total_flops = other.m_total_flops;
    this->m_total_trainable_parameters = other.m_total_trainable_parameters;
}

void RNN::copy(const RNN& other)
{
    this->m_input_shape     = other.m_input_shape;
    this->m_output_shape    = other.m_output_shape;

    this->m_hyperparameters = other.m_hyperparameters;
    this->m_parameters      = other.m_parameters;

    this->output            = other.output;
    this->required_output   = other.required_output;
    this->input             = other.input;

    this->layers            = other.layers;

    this->l_error           = other.l_error;
    this->l_output          = other.l_output;

    this->training_mode     = other.training_mode;
    this->minibatch_counter = other.minibatch_counter;
    this->minibatch_size    = other.minibatch_size;

    this->time_sequence_length  = other.time_sequence_length;
    this->time_step_idx         = other.time_step_idx;


    this->m_total_flops = other.m_total_flops;
    this->m_total_trainable_parameters = other.m_total_trainable_parameters;
}

Shape RNN::get_input_shape()
{
    return this->m_input_shape;
}

Shape RNN::get_output_shape()
{
    return this->m_output_shape;
}


void RNN::forward(Tensor &output, Tensor &input)
{
    //prepare layers for forward run
    for (unsigned int i = 0; i < layers.size(); i++)
        layers[i]->reset();
    time_step_idx = 0;

    //split input 4D tensor to 1D array of 3D tensors nn_input
    input.split_time_sequence(nn_input);

    //forward truth all time steps steps
    for (unsigned int t = 0; t < input.t(); t++)
    {
        //extract input
        l_output[t][0] = nn_input[t];

        //forward truth all layers
        for (unsigned int l = 0; l < layers.size(); l++)
        {
            layers[l]->forward(l_output[t][l + 1], l_output[t][l]);
        }

        nn_output[t] = l_output[t][layers.size()];
        time_step_idx++;
    }

    //RNN mode many to many
    if (output.t() == input.t())
    {
        output.concatenate_time_sequence(nn_output, time_step_idx);
    }
    else
    //RNN mode many to one, save only the last one
    {
        output = nn_output[time_sequence_length-1];
    }

    output.print();
}

void RNN::forward(std::vector<float> &output, std::vector<float> &input)
{
    this->input.set_from_host(input);

    forward(this->output, this->input);

    this->output.set_to_host(output);
}

void RNN::forward(std::vector<std::vector<float>> &output, std::vector<std::vector<float>> &input)
{
    for (unsigned int i = 0; i < output.size(); i++)
        forward(output[i], input[i]);
}


void RNN::train(Tensor &required_output, Tensor &input)
{
    forward(output, input);

    //RNN mode many to many
    if (required_output.t() == input.t())
    {
        this->error = required_output;
        this->error.sub(output);
    }
    //RNN mode many to one, compute error only from last
    else
    {
        this->error.clear();
    }

    /*
    unsigned int last_idx = l_output.size()-1;
    this->error = required_output;
    this->error.sub(l_output[last_idx]);

    train_from_error(this->error);
    */
}

void RNN::train_from_error(Tensor &error)
{
    /*
    bool update_weights = false;
    minibatch_counter++;

    if (minibatch_counter >= minibatch_size)
    {
        update_weights = true;
        minibatch_counter = 0;
    }

    unsigned int last_idx = layers.size()-1;
    l_error[last_idx + 1] = error;

    for (int i = last_idx; i>= 0; i--)
    {
        layers[i]->backward(l_error[i], l_error[i + 1], l_output[i], l_output[i + 1], update_weights);
    }
    */
}

Tensor& RNN::get_error_back()
{
    return l_error[0][0];
}

void RNN::train(std::vector<float> &required_output, std::vector<float> &input)
{
    this->required_output.set_from_host(required_output);
    this->input.set_from_host(input);

    train(this->required_output, this->input);
}

void RNN::train(std::vector<Tensor> &required_output, std::vector<Tensor> &input, unsigned int epoch_count, bool verbose)
{
    set_training_mode();

    unsigned int result_print_count = required_output.size()/100;
    if (result_print_count == 0)
        result_print_count = 1;

    for (unsigned int epoch = 0; epoch < epoch_count; epoch++)
    {
        auto indices = make_indices(required_output.size());

        for (unsigned int i = 0; i < required_output.size(); i++)
        {
            train(required_output[indices[i]], input[indices[i]]);

            if (verbose)
                if ((i%result_print_count) == 0)
                    std::cout << "epoch " << epoch+1 << " from " << epoch_count << " done = " << i*100.0/required_output.size() << "%" << "\n";
        }
    }

    unset_training_mode();
}


void RNN::set_training_mode()
{
    training_mode = true;

    for (unsigned int i = 0; i < layers.size(); i++)
        layers[i]->set_training_mode();
}

void RNN::unset_training_mode()
{
    training_mode = false;

    for (unsigned int i = 0; i < layers.size(); i++)
        layers[i]->unset_training_mode();
}

bool RNN::is_training_mode()
{
    return training_mode;
}

void RNN::reset()
{
    for (unsigned int i = 0; i < layers.size(); i++)
        layers[i]->reset();
}

void RNN::train(std::vector<std::vector<float>> &required_output, std::vector<std::vector<float>> &input, unsigned int epoch_count, bool verbose)
{
    set_training_mode();

    unsigned int result_print_count = required_output.size()/100;
    if (result_print_count == 0)
        result_print_count = 1;

    for (unsigned int epoch = 0; epoch < epoch_count; epoch++)
    {
        auto indices = make_indices(required_output.size());

        for (unsigned int i = 0; i < required_output.size(); i++)
        {
            train(required_output[indices[i]], input[indices[i]]);

            if (verbose)
                if ((i%result_print_count) == 0)
                    std::cout << "epoch " << epoch+1 << " from " << epoch_count << " done = " << i*100.0/required_output.size() << "%" << "\n";
        }
    }
    unset_training_mode();
}

void RNN::init(Json::Value json_config, Shape input_shape, Shape output_shape)
{
    this->m_input_shape       = input_shape;
    this->m_output_shape      = output_shape;
    this->training_mode       = false;
    this->minibatch_counter   = 0;

    this->time_step_idx = 0;
    this->time_sequence_length = input_shape.t();

    this->m_total_flops = 0;
    this->m_total_trainable_parameters = 0;

    if (json_config["network_log_file_name"] != Json::Value::null)
    {
        std::string network_log_file_name = json_config["network_log_file_name"].asString();
        if (network_log_file_name.size() > 0)
            network_log.set_output_file_name(network_log_file_name);
    }

    network_log << "network init start\n\n";

    if (input_shape.size() == 0)
    {
        this->m_input_shape.set(    json_config["input_shape"][0].asInt(),
                                    json_config["input_shape"][1].asInt(),
                                    json_config["input_shape"][2].asInt(),
                                    json_config["input_shape"][3].asInt());
    }

    if (output_shape.size() == 0)
    {
        this->m_output_shape.set(   json_config["output_shape"][0].asInt(),
                                    json_config["output_shape"][1].asInt(),
                                    json_config["output_shape"][2].asInt(),
                                    json_config["output_shape"][3].asInt() );
    }


    if (json_config["hyperparameters"]["learning_rate"] != Json::Value::null)
        m_hyperparameters["learning_rate"]     = json_config["hyperparameters"]["learning_rate"].asFloat();
    else
        m_hyperparameters["learning_rate"] = default_hyperparameters()["learning_rate"];

    if (json_config["hyperparameters"]["lambda1"] != Json::Value::null)
        m_hyperparameters["lambda1"]            = json_config["hyperparameters"]["lambda1"].asFloat();
    else
        m_hyperparameters["lambda1"] = default_hyperparameters()["lambda1"];

    if (json_config["hyperparameters"]["lambda2"] != Json::Value::null)
        m_hyperparameters["lambda2"]            = json_config["hyperparameters"]["lambda2"].asFloat();
    else
        m_hyperparameters["lambda2"] = default_hyperparameters()["lambda2"];


    if (json_config["hyperparameters"]["gradient_clip"] != Json::Value::null)
        m_hyperparameters["gradient_clip"]            = json_config["hyperparameters"]["gradient_clip"].asFloat();
    else
        m_hyperparameters["gradient_clip"] = default_hyperparameters()["gradient_clip"];

    if (json_config["hyperparameters"]["dropout"] != Json::Value::null)
        m_hyperparameters["dropout"]            = json_config["hyperparameters"]["dropout"].asFloat();
    else
        m_hyperparameters["dropout"] = default_hyperparameters()["dropout"];

    if (json_config["hyperparameters"]["minibatch_size"] != Json::Value::null)
        m_hyperparameters["minibatch_size"]  = json_config["hyperparameters"]["minibatch_size"].asInt();
    else
        m_hyperparameters["minibatch_size"]  = default_hyperparameters()["minibatch_size"];

    m_hyperparameters["time_sequence_length"]  = this->time_sequence_length;

    minibatch_size = m_hyperparameters["minibatch_size"].asInt();

    output.init(this->m_output_shape);
    required_output.init(this->m_output_shape);
    input.init(this->m_input_shape);
    error.init(this->m_output_shape);

    this->m_parameters["input_shape"][0]  = this->m_input_shape.w();
    this->m_parameters["input_shape"][1]  = this->m_input_shape.h();
    this->m_parameters["input_shape"][2]  = this->m_input_shape.d();
    this->m_parameters["input_shape"][3]  = this->m_input_shape.t();

    this->m_parameters["output_shape"][0] = this->m_output_shape.w();
    this->m_parameters["output_shape"][1] = this->m_output_shape.h();
    this->m_parameters["output_shape"][2] = this->m_output_shape.d();
    this->m_parameters["output_shape"][3] = this->m_output_shape.t();

    this->m_parameters["hyperparameters"] = m_hyperparameters;

    nn_input.resize(this->time_sequence_length);
    nn_output.resize(this->time_sequence_length);
    for (unsigned int t = 0; t < this->time_sequence_length; t++)
    {
        nn_input[t].init(this->m_input_shape.w(), this->m_input_shape.h(), this->m_input_shape.d());
        nn_output[t].init(this->m_output_shape.w(), this->m_output_shape.h(), this->m_output_shape.d());
    }

    l_error.resize(this->time_sequence_length);
    l_output.resize(this->time_sequence_length);

    for (unsigned int t = 0; t < this->time_sequence_length; t++)
    {
        l_error[t].push_back(Tensor(this->m_input_shape.w(), this->m_input_shape.h(), this->m_input_shape.d()));
        l_output[t].push_back(Tensor(this->m_input_shape.w(), this->m_input_shape.h(), this->m_input_shape.d()));
    }

    m_current_input_shape.set(this->m_input_shape.w(), this->m_input_shape.h(), this->m_input_shape.d());

    for (unsigned int i = 0; i < json_config["layers"].size(); i++)
    {
        std::string layer_type = json_config["layers"][i]["type"].asString();
        Shape shape;
        shape.set(
                    json_config["layers"][i]["shape"][0].asInt(),
                    json_config["layers"][i]["shape"][1].asInt(),
                    json_config["layers"][i]["shape"][2].asInt()
                 );

        std::string weights_file_name_prefix = json_config["layers"][i]["weights_file_name"].asString();

        add_layer(layer_type, shape, weights_file_name_prefix);

        m_current_input_shape = layers[layers.size()-1]->get_output_shape();
    }

    if (this->m_output_shape.size() == 0)
    {
        this->m_output_shape = m_current_input_shape;

        output.init(this->m_output_shape);
        required_output.init(this->m_output_shape);
        input.init(this->m_input_shape);
        error.init(this->m_output_shape);

        this->m_parameters["output_shape"][0] = this->m_output_shape.w();
        this->m_parameters["output_shape"][1] = this->m_output_shape.h();
        this->m_parameters["output_shape"][2] = this->m_output_shape.d();
        this->m_parameters["output_shape"][3] = this->m_output_shape.t();

    }




    network_log << "hyperparameters :\n";
    network_log << "learning_rate  = " << this->m_hyperparameters["learning_rate"].asFloat() << "\n";
    network_log << "lambda1        = " << this->m_hyperparameters["lambda1"].asFloat() << "\n";
    network_log << "lambda2        = " << this->m_hyperparameters["lambda2"].asFloat() << "\n";
    network_log << "gradient_clip  = " << this->m_hyperparameters["gradient_clip"].asFloat() << "\n";
    network_log << "dropout        = " << this->m_hyperparameters["dropout"].asFloat() << "\n";
    network_log << "minibatch_size = " << this->m_hyperparameters["minibatch_size"].asInt() << "\n";
    network_log << "time_sequence_length = " << this->m_hyperparameters["time_sequence_length"].asInt() << "\n";
    network_log << "\n\n";

    network_log << "input_shape  = " << this->m_input_shape.w() << " " << this->m_input_shape.h() << " " << this->m_input_shape.d() << " " << this->m_input_shape.t() << "\n";
    network_log << "output_shape = " << this->m_output_shape.w() << " " << this->m_output_shape.h() << " " << this->m_output_shape.d() << " " << this->m_output_shape.t() << "\n";
    network_log << "\n\n";

    network_log << "\nnetwork init done\n";
}


Shape RNN::add_layer(std::string layer_type, Shape shape, std::string weights_file_name_prefix)
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
    else
    if (layer_type == "output")
    {
        parameters["shape"][0] = m_output_shape.w();
        parameters["shape"][1] = m_output_shape.h();
        parameters["shape"][2] = m_output_shape.d();
        layer = new FCLayer(m_current_input_shape, parameters);
    }
    else
    if (layer_type == "recurrent")
    {
        layer = new RecurrentLayer(m_current_input_shape, parameters);
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
    if ((layer_type == "unpooling")||(layer_type == "un pooling"))
    {
        layer = new UnPoolingLayer(m_current_input_shape, parameters);
    }
    else
    if (layer_type == "dropout")
    {
        layer = new DropoutLayer(m_current_input_shape, parameters);
    }
    else
    if (layer_type == "crop")
    {
        layer = new CropLayer(m_current_input_shape, parameters);
    }
    else
    {
        std::cout << "ERROR : Unknow layer " << layer_type << "\n";
    }

    layers.push_back(layer);

    unsigned int layer_idx = layers.size()-1;

    output_shape = layers[layer_idx]->get_output_shape();

    for (unsigned int t = 0; t < time_sequence_length; t++)
    {
        l_error[t].push_back(Tensor(output_shape));
        l_output[t].push_back(Tensor(output_shape));
    }

    m_current_input_shape = output_shape;

    if (weights_file_name_prefix.size() > 0)
    {
        layer->load(weights_file_name_prefix);
    }

    this->m_parameters["layers"][layer_idx]["type"] = layer_type;

    this->m_parameters["layers"][layer_idx]["shape"][0] = shape.w();
    this->m_parameters["layers"][layer_idx]["shape"][1] = shape.h();
    this->m_parameters["layers"][layer_idx]["shape"][2] = shape.d();

    this->m_parameters["layers"][layer_idx]["input_shape"][0]  = layers[layer_idx]->get_input_shape().w();
    this->m_parameters["layers"][layer_idx]["input_shape"][1]  = layers[layer_idx]->get_input_shape().h();
    this->m_parameters["layers"][layer_idx]["input_shape"][2]  = layers[layer_idx]->get_input_shape().d();

    this->m_parameters["layers"][layer_idx]["output_shape"][0] = layers[layer_idx]->get_output_shape().w();
    this->m_parameters["layers"][layer_idx]["output_shape"][1] = layers[layer_idx]->get_output_shape().h();
    this->m_parameters["layers"][layer_idx]["output_shape"][2] = layers[layer_idx]->get_output_shape().d();

    m_total_flops+= layers[layer_idx]->get_flops();
    m_total_trainable_parameters+= layers[layer_idx]->get_trainable_parameters();

    return output_shape;
}

std::string RNN::asString()
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

void RNN::print()
{
    std::cout << asString() << "\n";
}

void RNN::save(std::string path)
{
    for (unsigned int i = 0; i < layers.size(); i++)
    {
        std::string weights_file_name_prefix = path + "layer_" + std::to_string(i);
        this->m_parameters["layers"][i]["weights_file_name"] = weights_file_name_prefix;
        layers[i]->save(weights_file_name_prefix);
    }

    JsonConfig json;
    json.result = this->m_parameters;

    json.save(path + "network_config.json");

    SVGVisualiser svg_visualiser(path + "network_config.json");

    svg_visualiser.process(path + "network.svg", m_input_shape);
}

void RNN::load_weights(std::string file_name_prefix)
{
    for (unsigned int layer = 0; layer < layers.size(); layer++)
    {
      std::string layer_file_name = file_name_prefix + "layer_" + std::to_string(layer);
      layers[layer]->load(layer_file_name);
    }
}

std::vector<unsigned int> RNN::make_indices(unsigned int count)
{
    std::vector<unsigned int> result(count);
    for (unsigned int i = 0; i < count; i++)
        result[i] = i;

    for (unsigned int i = 0; i < count; i++)
    {
        unsigned int idx_a = i;
        unsigned int idx_b = rand()%count;

        unsigned int tmp;

        tmp = result[idx_a];
        result[idx_a] = result[idx_b];
        result[idx_b] = tmp;
    }

    return result;
}

Json::Value RNN::default_hyperparameters(float learning_rate)
{
    Json::Value result;

    result["learning_rate"]  = learning_rate;
    result["lambda1"]        = learning_rate*0.001;
    result["lambda2"]        = learning_rate*0.001;
    result["gradient_clip"]  = 10.0;
    result["minibatch_size"] = 32;
    result["dropout"]        = 0.5;
    result["time_sequence_length"]        = 1;

    return result;
}


unsigned int RNN::get_layer_output_size()
{
    return l_output.size();
}

Tensor& RNN::get_layer_output(unsigned int layer_idx)
{
    return l_output[time_step_idx][layer_idx];
}

bool RNN::get_layer_weights_flag(unsigned int layer_idx)
{
    return layers[layer_idx]->has_weights();
}
