#include <experience_replay_buffer.h>
#include <iostream>

ExperienceReplayBuffer::ExperienceReplayBuffer()
{
    this->buffer_size = 0;
    this->current_ptr = 0;
}

ExperienceReplayBuffer::ExperienceReplayBuffer(ExperienceReplayBuffer& other)
{
    copy(other);
}

ExperienceReplayBuffer::ExperienceReplayBuffer(const ExperienceReplayBuffer& other)
{
    copy(other);
}

ExperienceReplayBuffer::ExperienceReplayBuffer(unsigned int buffer_size, unsigned int state_size, unsigned int actions_count)
{
    init(buffer_size, state_size, actions_count);
}

ExperienceReplayBuffer::~ExperienceReplayBuffer()
{

}

ExperienceReplayBuffer& ExperienceReplayBuffer::operator= (ExperienceReplayBuffer& other)
{
    copy(other);
    return *this;
}

ExperienceReplayBuffer& ExperienceReplayBuffer::operator= (const ExperienceReplayBuffer& other)
{
    copy(other);
    return *this;
}

void ExperienceReplayBuffer::copy(ExperienceReplayBuffer& other)
{
    this->state     = other.state;
    this->q_values  = other.q_values;
    this->action    = other.action;
    this->reward    = other.reward;
    this->terminal  = other.terminal;

    this->buffer_size  = other.buffer_size;
    this->current_ptr  = other.current_ptr;
}

void ExperienceReplayBuffer::copy(const ExperienceReplayBuffer& other)
{
    this->state     = other.state;
    this->q_values  = other.q_values;
    this->action    = other.action;
    this->reward    = other.reward;
    this->terminal  = other.terminal;

    this->buffer_size  = other.buffer_size;
    this->current_ptr  = other.current_ptr;
}

void ExperienceReplayBuffer::init(unsigned int buffer_size, unsigned int state_size, unsigned int actions_count)
{
    this->buffer_size = buffer_size;
    this->current_ptr = 0;

    this->state.resize(buffer_size);
    this->q_values.resize(buffer_size);
    this->action.resize(buffer_size);
    this->reward.resize(buffer_size);
    this->terminal.resize(buffer_size);


    for (unsigned int j = 0; j < this->buffer_size; j++)
    {
        this->state[j].resize(state_size);
        for (unsigned int i = 0; i < state_size; i++)
            this->state[j][i] = 0.0;
    }

    for (unsigned int j = 0; j < this->buffer_size; j++)
    {
        this->q_values[j].resize(actions_count);
        for (unsigned int i = 0; i < actions_count; i++)
            this->q_values[j][i] = 0.0;
    }

    for (unsigned int j = 0; j < this->buffer_size; j++)
        this->action[j] = 0;

    for (unsigned int j = 0; j < this->buffer_size; j++)
        this->reward[j] = 0.0;

    for (unsigned int j = 0; j < this->buffer_size; j++)
        this->terminal[j] = false;
}

bool ExperienceReplayBuffer::add(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float reward, bool terminal)
{
    if (current_ptr < buffer_size)
    {
        this->state[current_ptr]        = state;
        this->q_values[current_ptr]     = q_values;
        this->action[current_ptr]       = action;
        this->reward[current_ptr]       = reward;
        this->terminal[current_ptr]     = terminal;
        current_ptr++;

        return true;
    }
    else
    {
        return false;
    }
}

bool ExperienceReplayBuffer::is_full()
{
    if (current_ptr < buffer_size)
        return false;
    else
        return true;
}


void ExperienceReplayBuffer::compute(float gamma_value, float clamp_value)
{
    int move_idx = current_ptr - 2;

    for (unsigned int j = 0; j < size(); j++)
        reward[j] = clamp(reward[j], -1.0, 1.0);

    while (move_idx >= 0)
    {
        unsigned int action_idx = action[move_idx];

        float gamma = gamma_value;
        if (terminal[move_idx])
            gamma = 0.0;

        float q_new = reward[move_idx] + gamma*max(q_values[move_idx + 1]);

        q_new = clamp(q_new, -clamp_value, clamp_value);

        q_values[move_idx][action_idx] = q_new;

        move_idx--;
    }

    for (unsigned int j = 0; j < size(); j++)
    {
        for (unsigned int i = 0; i < q_values[j].size(); i++)
            q_values[j][i] = clamp(q_values[j][i], -clamp_value, clamp_value);
    }

    current_ptr = 0;
}

unsigned int ExperienceReplayBuffer::size()
{
    return q_values.size();
}


void ExperienceReplayBuffer::print()
{
    for (unsigned int j = 0; j < size(); j++)
    {
        std::cout << "s" << j << "\n";

        std::cout << "q_values = ";
        for (unsigned int i = 0; i < q_values[j].size(); i++)
            std::cout << q_values[j][i] << " ";
        std::cout << "\n";

        std::cout << "action   = " << action[j] << "\n";
        std::cout << "reward   = " << reward[j] << "\n";
        std::cout << "terminal = " << terminal[j] << "\n";

        std::cout << "\n\n\n\n";
    }
}


float ExperienceReplayBuffer::max(std::vector<float>& v)
{
    float result = v[0];

    for (unsigned int i = 0; i < v.size(); i++)
        if (v[i] > result)
            result = v[i];

    return result;
}

float ExperienceReplayBuffer::clamp(float value, float min, float max)
{
    if (value > max)
        value = max;

    if (value < min)
        value = min;

    return value;
}

std::vector<std::vector<float>>& ExperienceReplayBuffer::get_state()
{
    return this->state;
}

std::vector<std::vector<float>>& ExperienceReplayBuffer::get_q_values()
{
    return this->q_values;
}

std::vector<unsigned int>& ExperienceReplayBuffer::get_action()
{
    return this->action;
}

std::vector<float>& ExperienceReplayBuffer::get_reward()
{
    return this->reward;
}

std::vector<bool>& ExperienceReplayBuffer::get_terminal()
{
    return this->terminal;
}
