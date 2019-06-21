#include <dqn.h>
#include <iostream>

DQN::DQN()
{
    cnn = nullptr;
}

DQN::DQN(Shape state_shape, unsigned int actions_count, float gamma, unsigned int replay_buffer_size, std::string network_config_file_name)
{
    cnn = nullptr;
    init(state_shape, actions_count, gamma, replay_buffer_size, network_config_file_name);
}

DQN::DQN(DQN& other)
{
    copy(other);
}

DQN::DQN(const DQN& other)
{
    copy(other);
}

DQN::~DQN()
{
    if (cnn != nullptr)
    {
        delete cnn;
        cnn = nullptr;
    }
}

DQN& DQN::operator= (DQN& other)
{
    copy(other);
    return *this;
}

DQN& DQN::operator= (const DQN& other)
{
    copy(other);
    return *this;
}

void DQN::copy(DQN& other)
{
    this->state_shape   = other.state_shape;
    this->actions_count = other.actions_count;
    this->gamma         = other.gamma;

    this->q_values      = other.q_values;

    this->experience_replay_buffer = other.experience_replay_buffer;
}

void DQN::copy(const DQN& other)
{
    this->state_shape   = other.state_shape;
    this->actions_count = other.actions_count;
    this->gamma         = other.gamma;

    this->q_values      = other.q_values;

    this->experience_replay_buffer = other.experience_replay_buffer;
}

void DQN::init(Shape state_shape, unsigned int actions_count, float gamma, unsigned int replay_buffer_size, std::string network_config_file_name)
{
    this->state_shape   = state_shape;
    this->actions_count = actions_count;
    this->gamma         = gamma;

    this->q_values.resize(actions_count);

    this->experience_replay_buffer.init(replay_buffer_size, this->state_shape.size(), this->actions_count);

    Shape output_shape(1, 1, this->actions_count);

    if (cnn != nullptr)
    {
        delete cnn;
        cnn = nullptr;
    }

    if (network_config_file_name != "")
    {
        cnn = new CNN(network_config_file_name, this->state_shape, output_shape);
    }
    else
    {
        cnn = new CNN(this->state_shape, output_shape);
    }
}

void DQN::add_layer(std::string layer_type, Shape shape)
{
    cnn->add_layer(layer_type, shape);
}


std::vector<float>& DQN::forward(std::vector<float> &state)
{
    cnn->forward(q_values, state);
    return q_values;
}

std::vector<float>& DQN::get_q_values()
{
    return q_values;
}

bool DQN::add(std::vector<float>& state, std::vector<float>& q_values, unsigned int action, float reward, bool terminal)
{
    return experience_replay_buffer.add(state, q_values, action, reward, terminal);
}

bool DQN::is_full()
{
    return experience_replay_buffer.is_full();
}

void DQN::train()
{
    if (experience_replay_buffer.is_full() == false)
        return;

    experience_replay_buffer.compute(gamma);

    cnn->train(experience_replay_buffer.get_q_values(), experience_replay_buffer.get_state(), 1, false);
}

void DQN::print()
{
    cnn->print();
}

void DQN::print_buffer()
{
    experience_replay_buffer.print();
}


void DQN::save(std::string path)
{
    cnn->save(path);
}

void DQN::load_weights(std::string file_name_prefix)
{
    cnn->load_weights(file_name_prefix);
}
