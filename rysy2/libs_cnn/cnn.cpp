#include <cnn.h>

#include <json_config.h>

CNN::CNN()
{
    m_hyperparameters["learning_rate"]  = 0.001;
    m_hyperparameters["lambda1"]        = 0.0;
    m_hyperparameters["lambda2"]        = 0.0;
    m_hyperparameters["dropout"]        = 0.0;
    m_hyperparameters["minibatch_size"] = 32;
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

CNN::~CNN()
{

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
    this->m_input_shape = other.m_input_shape;
    this->m_output_shape = other.m_output_shape;

    this->m_hyperparameters = other.m_hyperparameters;

    this->output            = other.output;
    this->required_output   = other.required_output;
    this->input             = other.input;

    this->layers            = other.layers;

    this->l_error           = other.l_error;
    this->l_output          = other.l_output;
}

void CNN::copy(const CNN& other)
{
    this->m_input_shape = other.m_input_shape;
    this->m_output_shape = other.m_output_shape;

    this->m_hyperparameters = other.m_hyperparameters;

    this->output            = other.output;
    this->required_output   = other.required_output;
    this->input             = other.input;

    this->layers            = other.layers;

    this->l_error           = other.l_error;
    this->l_output          = other.l_output;
}



void CNN::forward(Tensor &output, Tensor &input)
{

}

void CNN::forward(std::vector<float> &output, std::vector<float> &input)
{
    this->output.clear();
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
    for (unsigned int i = 0; i < required_output.size(); i++)
        train(required_output[i], input[i]);
}

void CNN::train(std::vector<std::vector<float>> &required_output, std::vector<std::vector<float>> &input)
{
    for (unsigned int i = 0; i < required_output.size(); i++)
        train(required_output[i], input[i]);
}

void CNN::init(Json::Value json_config, Shape input_shape, Shape output_shape)
{
    this->m_input_shape     = input_shape;
    this->m_output_shape    = output_shape;

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
        m_hyperparameters["learning_rate"] = 0.001;

    if (json_config["hyperparameters"]["lambda1"] != Json::Value::null)
        m_hyperparameters["lambda1"]            = json_config["hyperparameters"]["lambda1"].asFloat();
    else
        m_hyperparameters["lambda1"] = 0.0;

    if (json_config["hyperparameters"]["lambda2"] != Json::Value::null)
        m_hyperparameters["lambda2"]            = json_config["hyperparameters"]["lambda2"].asFloat();
    else
        m_hyperparameters["lambda2"] = 0.0;

    if (json_config["hyperparameters"]["dropout"] != Json::Value::null)
        m_hyperparameters["dropout"]            = json_config["hyperparameters"]["dropout"].asFloat();
    else
        m_hyperparameters["dropout"] = 0.0;

    if (json_config["hyperparameters"]["minibatch_size"] != Json::Value::null)
        m_hyperparameters["minibatch_size"]  = json_config["hyperparameters"]["minibatch_size"].asInt();
    else
        m_hyperparameters["minibatch_size"]  = 32;

    network_log << "hyperparameters :\n";
    network_log << "learning_rate = " << m_hyperparameters["learning_rate"].asFloat() << "\n";
    network_log << "lambda1 = " << m_hyperparameters["lambda1"].asFloat() << "\n";
    network_log << "lambda2 = " << m_hyperparameters["lambda2"].asFloat() << "\n";
    network_log << "dropout = " << m_hyperparameters["dropout"].asFloat() << "\n";
    network_log << "minibatch_size = " << m_hyperparameters["minibatch_size"].asInt() << "\n";
    network_log << "\n\n";

    network_log << "input_shape = " << m_input_shape.w() << " " << m_input_shape.h() << " " << m_input_shape.d() << "\n";
    network_log << "output_shape = " << m_output_shape.w() << " " << m_output_shape.h() << " " << m_output_shape.d() << "\n";
    network_log << "\n\n";

}
