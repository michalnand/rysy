#include <dqrn.h>
#include <iostream>

DQRN::DQRN()
{
    rnn = nullptr;
    activity = nullptr;
}

DQRN::DQRN(Shape state_shape, unsigned int actions_count, float gamma, unsigned int replay_buffer_size, std::string network_config_file_name)
{
    rnn = nullptr;
    activity = nullptr;
    init(state_shape, actions_count, gamma, replay_buffer_size, network_config_file_name);
}

DQRN::DQRN(DQRN& other)
{
    copy(other);
}

DQRN::DQRN(const DQRN& other)
{
    copy(other);
}

DQRN::~DQRN()
{
    if (rnn != nullptr)
    {
        delete rnn;
        rnn = nullptr;
    }

    if (activity != nullptr)
    {
        delete activity;
        activity = nullptr;
    }
}

DQRN& DQRN::operator= (DQRN& other)
{
    copy(other);
    return *this;
}

DQRN& DQRN::operator= (const DQRN& other)
{
    copy(other);
    return *this;
}

void DQRN::copy(DQRN& other)
{
    this->state_shape   = other.state_shape;
    this->actions_count = other.actions_count;
    this->gamma         = other.gamma;

    this->q_values      = other.q_values;

    this->experience_replay_buffer = other.experience_replay_buffer;
}

void DQRN::copy(const DQRN& other)
{
    this->state_shape   = other.state_shape;
    this->actions_count = other.actions_count;
    this->gamma         = other.gamma;

    this->q_values      = other.q_values;

    this->experience_replay_buffer = other.experience_replay_buffer;
}

void DQRN::init(Shape state_shape, unsigned int actions_count, float gamma, unsigned int replay_buffer_size, std::string network_config_file_name)
{
    this->state_shape   = state_shape;
    this->actions_count = actions_count;
    this->gamma         = gamma;

    this->q_values.resize(actions_count);

    this->experience_replay_buffer.init(replay_buffer_size, this->state_shape.size(), this->actions_count);

    v_state.resize(state_shape.size());

    Shape output_shape(1, 1, this->actions_count);

    if (rnn != nullptr)
    {
        delete rnn;
        rnn = nullptr;
    }

    if (network_config_file_name != "")
    {
        rnn = new RNN(network_config_file_name, this->state_shape, output_shape);
    }
    else
    {
        rnn = new RNN(this->state_shape, output_shape);
    }

    //TODO
    //activity = new NetworkActivity(*rnn);
}

void DQRN::add_layer(std::string layer_type, Shape shape)
{
    rnn->add_layer(layer_type, shape);
}


std::vector<float>& DQRN::forward(std::vector<float> &state)
{
    rnn->forward(q_values, state);
    return q_values;
}

std::vector<float>& DQRN::forward(float *state)
{
    for (unsigned int i = 0; i < v_state.size(); i++)
        v_state[i] = state[i];

    rnn->forward(q_values, v_state);
    return q_values;
}

std::vector<float>& DQRN::get_q_values()
{
    return q_values;
}

bool DQRN::add(std::vector<float>& state, std::vector<float>& q_values, unsigned int action, float reward, bool terminal)
{
    return experience_replay_buffer.add(state, q_values, action, reward, terminal);
}

bool DQRN::add(float *state, std::vector<float>& q_values, unsigned int action, float reward, bool terminal)
{
    for (unsigned int i = 0; i < v_state.size(); i++)
        v_state[i] = state[i];

    return experience_replay_buffer.add(v_state, q_values, action, reward, terminal);
}

bool DQRN::is_full()
{
    return experience_replay_buffer.is_full();
}

void DQRN::train()
{
    if (experience_replay_buffer.is_full() == false)
        return;

    experience_replay_buffer.compute(gamma);

    rnn->train(experience_replay_buffer.get_q_values(), experience_replay_buffer.get_state(), 1, false);
}

void DQRN::print()
{
    rnn->print();
}

void DQRN::print_buffer()
{
    experience_replay_buffer.print();
}


void DQRN::save(std::string path)
{
    rnn->save(path);
}

void DQRN::load_weights(std::string file_name_prefix)
{
    rnn->load_weights(file_name_prefix);
}

void DQRN::add_activity_map()
{
    activity->add();
}

void DQRN::save_activity_map(std::string path)
{
    activity->save(path);
}
