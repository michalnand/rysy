#include <cnn.h>

#include <iostream>
#include <json_config.h>

#include <layers/activation_elu_layer.h>
#include <layers/activation_relu_layer.h>

#include <layers/convolution_layer.h>
#include <layers/dense_convolution_layer.h>
#include <layers/fc_layer.h>


#include <layers/max_pooling_layer.h>
#include <layers/average_pooling_layer.h>
#include <layers/unpooling_layer.h>

#include <layers/dropout_layer.h>
#include <layers/crop_layer.h>
#include <layers/flatten_layer.h>

#include <layers/highway_block_layer.h>


#include <svg_visualiser.h>
#include <image_save.h>
#include <utils.h>


#include <kernels/solver_adam.cuh>


CNN::CNN()
{
    this->m_hyperparameters = default_hyperparameters();


    this->training_mode       = false;
    this->minibatch_counter   = 0;
    this->minibatch_size      = 32;

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

CNN::CNN(std::string network_config_file_name, Shape input_shape, Shape output_shape)
{
    JsonConfig json(network_config_file_name);
    init(json.result, input_shape, output_shape);
}

CNN::CNN(Json::Value json_config, Shape input_shape, Shape output_shape)
{
    init(json_config, input_shape, output_shape);
}

CNN::CNN(Shape input_shape, Shape output_shape, float learning_rate, float lambda1, float lambda2, float gradient_clip, float dropout, unsigned int minibatch_size)
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


    this->m_total_flops = other.m_total_flops;
    this->m_total_trainable_parameters = other.m_total_trainable_parameters;
}

void CNN::copy(const CNN& other)
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

    this->m_total_flops = other.m_total_flops;
    this->m_total_trainable_parameters = other.m_total_trainable_parameters;
}

Shape CNN::get_input_shape()
{
    return this->m_input_shape;
}

Shape CNN::get_output_shape()
{
    return this->m_output_shape;
}

void CNN::forward(Tensor &output, Tensor &input)
{
    l_output[0] = input;

    for (unsigned int i = 0; i < layers.size(); i++)
    {
        layers[i]->forward(l_output[i+1], l_output[i]);
    }

    output = l_output[layers.size()];
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
    unsigned int last_idx = l_output.size()-1;
    l_output[0] = input;

    forward(l_output[last_idx], l_output[0]);

    this->error = required_output;
    this->error.sub(l_output[last_idx]);

    train_from_error(this->error);
}

void CNN::train_from_error(Tensor &error)
{
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
}

Tensor& CNN::get_error_back()
{
    return l_error[0];
}


std::vector<float> CNN::kernel_visualisation(unsigned int layer, unsigned int kernel)
{
    Shape output_shape = layers[layer]->get_output_shape();
    Shape input_shape  = m_input_shape;

    Tensor t_result(input_shape);
    t_result.set_random(0.001);


    std::vector<float> v_output(output_shape.size());
    std::vector<float> v_target(output_shape.size());
    std::vector<float> v_error(output_shape.size());


    for (unsigned int iteration = 0; iteration < 256; iteration++)
    {
        for (unsigned int i = 0; i < l_error.size(); i++)
            l_error[i].clear();
        for (unsigned int i = 0; i < l_output.size(); i++)
            l_output[i].clear();

        l_output[0] = t_result;

        for (unsigned int i = 0; i <= layer; i++)
        {
            layers[i]->forward(l_output[i + 1], l_output[i]);
        }

        l_output[layer + 1].set_to_host(v_output);

        for (unsigned int i = 0; i < v_error.size(); i++)
            v_error[i] = 0.0;

        for (unsigned int y = 0; y < output_shape.h(); y++)
            for (unsigned int x = 0; x < output_shape.w(); x++)
            {
                unsigned int idx = (kernel*output_shape.h() + y)*output_shape.w() + x;
                v_error[idx] = v_output[idx];
            }

        float length = 0.0;
        for (unsigned int i = 0; i < v_error.size(); i++)
                length+= v_error[i]*v_error[i];
        length = sqrt(length);

        for (unsigned int i = 0; i < v_error.size(); i++)
            v_error[i] = v_error[i]/(length + 0.0001);



        l_error[layer + 1].set_from_host(v_error);


        for (int i = layer; i>= 0; i--)
        {
            layers[i]->backward(l_error[i], l_error[i + 1], l_output[i], l_output[i + 1], false, false);
        }


        Tensor tmp = t_result;
        tmp.mul(0.01);
        t_result.add(l_error[0]);
        t_result.sub(tmp);

        Tensor t_noise(t_result.shape());
        t_noise.set_random(0.001);
        t_result.add(t_noise);
    }


    std::vector<float> v_result(t_result.size());
    t_result.set_to_host(v_result);
    normalise(v_result, 0.0, 1.0);

    return v_result;
}

void CNN::kernel_visualisation(std::string image_path)
{
    unsigned int spacing = 4;
    bool grayscale;
    if (m_input_shape.d() == 1)
        grayscale = true;
    else
        grayscale = false;

    unsigned int channels = 0;
    if (grayscale)
        channels = 1;
    else
        channels = 3;


    for (unsigned int layer = 0; layer < layers.size(); layer++)
        if (layers[layer]->has_weights())
            //if (layers[layer]->get_output_shape().w() > 1 && layers[layer]->get_output_shape().h() > 1)
            {
                unsigned int kernels_count =  layers[layer]->get_output_shape().d();


                auto rectangle = make_rectangle(kernels_count);
                unsigned int input_width  = m_input_shape.w();
                unsigned int input_height = m_input_shape.h();
                unsigned int output_width  = rectangle.x*(input_width + spacing);
                unsigned int output_height = rectangle.y*(input_height + spacing);

                ImageSave image(output_width, output_height, grayscale);

                std::vector<float> result(output_width*output_height*channels);
                for (unsigned int i = 0; i < result.size(); i++)
                    result[i] = 0.0;

                JsonConfig json;

                json.result["shape"][0] = input_width;
                json.result["shape"][1] = input_height;
                json.result["shape"][2] = m_input_shape.d();
                json.result["kernels_count"] = kernels_count;

                for (unsigned int kernel = 0; kernel < kernels_count; kernel++)
                {
                    auto kernel_result = kernel_visualisation(layer, kernel);

                    unsigned int x0 = (kernel%rectangle.x)*(input_width + spacing) + spacing/2;
                    unsigned int y0 = (kernel/rectangle.x)*(input_height + spacing) + spacing/2;

                    for (unsigned int ch = 0; ch < channels; ch++)
                        for (unsigned int y = 0; y < input_height; y++)
                            for (unsigned int x = 0; x < input_width; x++)
                            {
                                unsigned int input_idx  = (ch*input_height + y)*input_width + x;
                                unsigned int output_idx = (ch*output_height + y + y0)*output_width + x + x0;
                                result[output_idx] = kernel_result[input_idx];
                            }

                    for (unsigned int ch = 0; ch < m_input_shape.d(); ch++)
                        for (unsigned int y = 0; y < input_height; y++)
                            for (unsigned int x = 0; x < input_width; x++)
                            {
                                unsigned int input_idx  = (ch*input_height + y)*input_width + x;
                                json.result["result"][kernel][ch][y][x] = kernel_result[input_idx];
                            }
                }

                std::string image_file_name = image_path + std::to_string(layer) + ".png";
                image.save(image_file_name, result);

                std::string json_file_name = image_path + std::to_string(layer) + ".json";
                json.save(json_file_name);
            }
}

void CNN::activity_visualisation(std::string image_path, std::vector<float> &input_)
{
    input.set_from_host(input_);
    forward(output, input);

    std::vector<unsigned int> layer_output_list;

    layer_output_list.push_back(0);
    for (unsigned int layer = 0; layer < layers.size(); layer++)
        if (layers[layer]->is_activation() == true)
            if (layers[layer]->get_output_shape().w() > 1 && layers[layer]->get_output_shape().h() > 1)
                layer_output_list.push_back(layer + 1);

    unsigned int spacing = 4;

    for (unsigned int i = 0; i < layer_output_list.size(); i++)
    {
        unsigned int layer = layer_output_list[i];

        unsigned int width         = l_output[layer].shape().w();
        unsigned int height        = l_output[layer].shape().h();
        unsigned int kernels_count = l_output[layer].shape().d();

        auto rectangle = make_rectangle(kernels_count);


        unsigned int output_width  = rectangle.x*(width  + spacing);
        unsigned int output_height = rectangle.y*(height + spacing);


        std::vector<float> layer_output(l_output[layer].size());
        l_output[layer].set_to_host(layer_output);

        std::vector<float> result(output_width*output_height);
        for (unsigned int i = 0; i < result.size(); i++)
            result[i] = 0.0;

        for (unsigned int kernel = 0; kernel < kernels_count; kernel++)
        {
            unsigned int x0 = (kernel%rectangle.x)*(width  + spacing) + spacing/2;
            unsigned int y0 = (kernel/rectangle.x)*(height + spacing) + spacing/2;

            for (unsigned int y = 0; y < height; y++)
                for (unsigned int x = 0; x < width; x++)
                {
                    unsigned int input_idx  = (kernel*height + y)*width + x;
                    unsigned int output_idx = (y + y0)*output_width + x + x0;
                    float v = -1.0*layer_output[input_idx];
                    result[output_idx] = v;
                }
        }


        std::string image_file_name;

        if (layer == 0)
        {
            image_file_name = image_path + "0_input" + ".png";
        }
        else
        {
            image_file_name = image_path + std::to_string(layer-1) + ".png";
        }

        ImageSave image(output_width, output_height, true);
        image.save(image_file_name, result);
    }
}







void CNN::heatmap_visualisation(std::string image_path, std::vector<float> &input_)
{
    input.set_from_host(input_);
    forward(output, input);

    unsigned int layer_idx = 0;
    for (unsigned int layer = 0; layer < layers.size(); layer++)
        if (layers[layer]->is_activation() == true)
            if (layers[layer]->get_output_shape().w() > 1 && layers[layer]->get_output_shape().h() > 1)
                layer_idx = layer;



    std::vector<float> layer_output(l_output[layer_idx].size());
    l_output[layer_idx].set_to_host(layer_output);

    unsigned int layer_width  = l_output[layer_idx].shape().w();
    unsigned int layer_height = l_output[layer_idx].shape().h();
    unsigned int layer_depth  = l_output[layer_idx].shape().d();

    std::vector<std::vector<float>> activity_sum;

    activity_sum.resize(layer_height);
    for (unsigned int y = 0; y < layer_height; y++)
    {
        activity_sum[y].resize(layer_width);
        for (unsigned int x = 0; x < layer_width; x++)
            activity_sum[y][x] = 0.0;
    }

    for (unsigned int ch = 0; ch < layer_depth; ch++)
        for (unsigned int y = 0; y < layer_height; y++)
            for (unsigned int x = 0; x < layer_width; x++)
            {
                unsigned int idx = (ch*layer_height + y)*layer_width + x;
                activity_sum[y][x]+= layer_output[idx];
            }

    unsigned int output_width         = m_input_shape.w();
    unsigned int output_height        = m_input_shape.h();

    unsigned int scaling_y = output_height/layer_height;
    unsigned int scaling_x = output_width/layer_width;

    std::vector<std::vector<float>> activity_sum_upscaled;

    activity_sum_upscaled.resize(output_height);
    for (unsigned int y = 0; y < output_height; y++)
    {
        activity_sum_upscaled[y].resize(output_width);
        for (unsigned int x = 0; x < output_width; x++)
            activity_sum_upscaled[y][x] = 0.0;
    }

    for (unsigned int y = 0; y < layer_height - 1; y++)
        for (unsigned int x = 0; x < layer_width - 1; x++)
        {
            float x00 = activity_sum[y + 0][x + 0];
            float x01 = activity_sum[y + 0][x + 1];
            float x10 = activity_sum[y + 1][x + 0];
            float x11 = activity_sum[y + 1][x + 1];

            for (unsigned int ky = 0; ky <= scaling_y; ky++)
                for (unsigned int kx = 0; kx <= scaling_x; kx++)
                {
                    float v = 0.0;
                    v+= (scaling_x - kx)*(scaling_y - ky)*x00;
                    v+= (kx)*(scaling_y - ky)*x01;

                    v+= (scaling_x - kx)*(ky)*x10;
                    v+= (kx)*(ky)*x11;

                    activity_sum_upscaled[y*scaling_y + ky][x*scaling_x + kx] = v;
                }

        }

    std::string image_file_name = image_path + "_heatmap.png";
    ImageSave image(output_width, output_height, true);
    image.save(image_file_name, activity_sum_upscaled);
}

void CNN::train(std::vector<float> &required_output, std::vector<float> &input)
{
    this->required_output.set_from_host(required_output);
    this->input.set_from_host(input);

    train(this->required_output, this->input);
}

void CNN::train(std::vector<Tensor> &required_output, std::vector<Tensor> &input, unsigned int epoch_count, bool verbose)
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

void CNN::train(std::vector<std::vector<float>> &required_output, std::vector<std::vector<float>> &input, unsigned int epoch_count, bool verbose)
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

    if (input_shape.size() == 0)
    {
        this->m_input_shape.set(    json_config["input_shape"][0].asInt(),
                                    json_config["input_shape"][1].asInt(),
                                    json_config["input_shape"][2].asInt() );
    }

    if (output_shape.size() == 0)
    {
        this->m_output_shape.set(   json_config["output_shape"][0].asInt(),
                                    json_config["output_shape"][1].asInt(),
                                    json_config["output_shape"][2].asInt() );
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




    minibatch_size = m_hyperparameters["minibatch_size"].asInt();

    output.init(this->m_output_shape);
    required_output.init(this->m_output_shape);
    input.init(this->m_input_shape);
    error.init(this->m_output_shape);

    this->m_parameters["input_shape"][0]  = this->m_input_shape.w();
    this->m_parameters["input_shape"][1]  = this->m_input_shape.h();
    this->m_parameters["input_shape"][2]  = this->m_input_shape.d();
    this->m_parameters["output_shape"][0] = this->m_output_shape.w();
    this->m_parameters["output_shape"][1] = this->m_output_shape.h();
    this->m_parameters["output_shape"][2] = this->m_output_shape.d();

    this->m_parameters["hyperparameters"] = m_hyperparameters;

    l_error.push_back(Tensor(this->m_input_shape));
    l_output.push_back(Tensor(this->m_input_shape));

    m_current_input_shape = this->m_input_shape;

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
    }


    network_log << "hyperparameters :\n";
    network_log << "learning_rate  = " << this->m_hyperparameters["learning_rate"].asFloat() << "\n";
    network_log << "lambda1        = " << this->m_hyperparameters["lambda1"].asFloat() << "\n";
    network_log << "lambda2        = " << this->m_hyperparameters["lambda2"].asFloat() << "\n";
    network_log << "gradient_clip  = " << this->m_hyperparameters["gradient_clip"].asFloat() << "\n";
    network_log << "dropout        = " << this->m_hyperparameters["dropout"].asFloat() << "\n";
    network_log << "minibatch_size = " << this->m_hyperparameters["minibatch_size"].asInt() << "\n";
    network_log << "\n\n";

    network_log << "input_shape  = " << this->m_input_shape.w() << " " << this->m_input_shape.h() << " " << this->m_input_shape.d() << "\n";
    network_log << "output_shape = " << this->m_output_shape.w() << " " << this->m_output_shape.h() << " " << this->m_output_shape.d() << "\n";
    network_log << "\n\n";

    network_log << "\nnetwork init done\n";
}


Shape CNN::add_layer(std::string layer_type, Shape shape, std::string weights_file_name_prefix)
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
    if (layer_type == "flatten")
    {
        layer = new FlattenLayer(m_current_input_shape, parameters);
    }
    else
    if (layer_type == "highway")
    {
        layer = new HighwayBlockLayer(m_current_input_shape, parameters);
    }
    else
    {
        std::cout << "ERROR : Unknow layer " << layer_type << "\n";
    }

    layers.push_back(layer);

    unsigned int layer_idx = layers.size()-1;

    output_shape = layers[layer_idx]->get_output_shape();

    l_error.push_back(Tensor(output_shape));
    l_output.push_back(Tensor(output_shape));

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

void CNN::print()
{
    std::cout << asString() << "\n";
}

void CNN::save(std::string path)
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

void CNN::load_weights(std::string file_name_prefix)
{
    for (unsigned int layer = 0; layer < layers.size(); layer++)
    {
      std::string layer_file_name = file_name_prefix + "layer_" + std::to_string(layer);
      layers[layer]->load(layer_file_name);
    }
}

std::vector<unsigned int> CNN::make_indices(unsigned int count)
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

Json::Value CNN::default_hyperparameters(float learning_rate)
{
    Json::Value result;

    result["learning_rate"]  = learning_rate;
    result["lambda1"]        = learning_rate*0.001;
    result["lambda2"]        = learning_rate*0.001;
    result["gradient_clip"]  = 10.0;
    result["minibatch_size"] = 32;
    result["dropout"]        = 0.5;

    return result;
}

unsigned int CNN::get_layers_count()
{
    return layers.size();
}

unsigned int CNN::get_layer_output_size()
{
    return l_output.size();
}

Tensor& CNN::get_layer_output(unsigned int layer_idx)
{
    return l_output[layer_idx];
}

bool CNN::get_layer_weights_flag(unsigned int layer_idx)
{
    return layers[layer_idx]->has_weights();
}

bool CNN::get_layer_activation_flag(unsigned int layer_idx)
{
    return layers[layer_idx]->is_activation();
}
