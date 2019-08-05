#include <dqn_curiosity.h>
#include <iostream>

DQNCuriosity::DQNCuriosity()
{
    cnn = nullptr;
    icm = nullptr;
}

DQNCuriosity::DQNCuriosity(Shape state_shape, unsigned int actions_count, float gamma, float curiosity_ratio, unsigned int replay_buffer_size, std::string network_config_path)
{
    cnn = nullptr;
    icm = nullptr;
    init(state_shape, actions_count, gamma, curiosity_ratio, replay_buffer_size, network_config_path);
}

DQNCuriosity::DQNCuriosity(DQNCuriosity& other)
{
    copy(other);
}

DQNCuriosity::DQNCuriosity(const DQNCuriosity& other)
{
    copy(other);
}

DQNCuriosity::~DQNCuriosity()
{
    if (cnn != nullptr)
    {
        delete cnn;
        cnn = nullptr;
    }

    if (icm != nullptr)
    {
        delete icm;
        icm = nullptr;
    }
}

DQNCuriosity& DQNCuriosity::operator= (DQNCuriosity& other)
{
    copy(other);
    return *this;
}

DQNCuriosity& DQNCuriosity::operator= (const DQNCuriosity& other)
{
    copy(other);
    return *this;
}

void DQNCuriosity::copy(DQNCuriosity& other)
{
    this->state_shape   = other.state_shape;
    this->actions_count = other.actions_count;
    this->gamma         = other.gamma;

    this->q_values      = other.q_values;

    this->experience_replay_buffer = other.experience_replay_buffer;
}

void DQNCuriosity::copy(const DQNCuriosity& other)
{
    this->state_shape   = other.state_shape;
    this->actions_count = other.actions_count;
    this->gamma         = other.gamma;

    this->q_values      = other.q_values;

    this->experience_replay_buffer = other.experience_replay_buffer;
}

void DQNCuriosity::init(Shape state_shape, unsigned int actions_count, float gamma, float curiosity_ratio, unsigned int replay_buffer_size, std::string network_config_path)
{
    this->state_shape       = state_shape;
    this->actions_count     = actions_count;
    this->gamma             = gamma;
    this->curiosity_ratio   = curiosity_ratio;

    this->q_values.resize(actions_count);

    this->experience_replay_buffer.init(replay_buffer_size, this->state_shape.size(), this->actions_count);

    v_state.resize(state_shape.size());

    Shape output_shape(1, 1, this->actions_count);

    cnn = new CNN(network_config_path + "dqn/network_config.json", this->state_shape, output_shape);

    icm = new ICM(state_shape, actions_count, network_config_path);
}

std::vector<float>& DQNCuriosity::forward(std::vector<float> &state)
{
    cnn->forward(q_values, state);
    return q_values;
}

std::vector<float>& DQNCuriosity::forward(float *state)
{
    for (unsigned int i = 0; i < v_state.size(); i++)
        v_state[i] = state[i];

    cnn->forward(q_values, v_state);
    return q_values;
}

std::vector<float>& DQNCuriosity::get_q_values()
{
    return q_values;
}

bool DQNCuriosity::add(std::vector<float>& state, std::vector<float>& q_values, unsigned int action, float reward, bool terminal)
{
    return experience_replay_buffer.add(state, q_values, action, reward, terminal);
}

bool DQNCuriosity::add(float *state, std::vector<float>& q_values, unsigned int action, float reward, bool terminal)
{
    for (unsigned int i = 0; i < v_state.size(); i++)
        v_state[i] = state[i];

    return experience_replay_buffer.add(v_state, q_values, action, reward, terminal);
}

bool DQNCuriosity::is_full()
{
    return experience_replay_buffer.is_full();
}

void DQNCuriosity::train()
{
    if (experience_replay_buffer.is_full() == false)
        return;

    icm->train(experience_replay_buffer);

    std::cout << "inverse loss " << icm->get_icm_result().inverse_loss << "\n";
    std::cout << "forward loss " << icm->get_icm_result().forward_loss << "\n";
    std::cout << "forward classification " << icm->get_icm_result().inverse_classification_success << "\n";
    std::cout << "\n";

    experience_replay_buffer.compute(gamma, 10.0, curiosity_ratio);

    cnn->train(experience_replay_buffer.get_q_values(), experience_replay_buffer.get_state(), 1, false);
}

void DQNCuriosity::print()
{
    cnn->print();
    icm->print();
}

void DQNCuriosity::print_buffer()
{
    experience_replay_buffer.print();
}


void DQNCuriosity::save(std::string path)
{
    cnn->save(path + "dqn/trained/");
    icm->save(path);
}


void DQNCuriosity::load_weights(std::string path)
{
    cnn->load_weights(path + "dqn/trained/");
    icm->load(path);
}

sICMResult DQNCuriosity::get_icm_result()
{
    return icm->get_icm_result();
}
