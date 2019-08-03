#ifndef _EXPERIENCE_REPLAY_BUFFER_H_
#define _EXPERIENCE_REPLAY_BUFFER_H_

#include <vector>

class ExperienceReplayBuffer
{
    public:
        ExperienceReplayBuffer();
        ExperienceReplayBuffer(ExperienceReplayBuffer& other);
        ExperienceReplayBuffer(const ExperienceReplayBuffer& other);

        ExperienceReplayBuffer(unsigned int buffer_size, unsigned int state_size, unsigned int actions_count);

        virtual ~ExperienceReplayBuffer();

        ExperienceReplayBuffer& operator= (ExperienceReplayBuffer& other);
        ExperienceReplayBuffer& operator= (const ExperienceReplayBuffer& other);

    protected:
        void copy(ExperienceReplayBuffer& other);
        void copy(const ExperienceReplayBuffer& other);

    public:

        void init(unsigned int buffer_size, unsigned int state_size, unsigned int actions_count);

        bool add(std::vector<float> &state, std::vector<float> &q_values, unsigned int action, float reward, bool terminal = false);
        bool is_full();

        void compute(float gamma = 0.99, float clamp_value = 10.0, float curiosity_ratio = 0.0);

        void set_curiosity(unsigned int idx, float curiosity);

        unsigned int size();
        void print();

    protected:
        float max(std::vector<float>& v);
        float clamp(float value, float min, float max);

    public:
        std::vector<std::vector<float>>&    get_state();
        std::vector<std::vector<float>>&    get_q_values();
        std::vector<unsigned int>&          get_action();
        std::vector<float>&                 get_reward();
        std::vector<float>&                 get_curiosity();
        std::vector<bool>&                  get_terminal();

    protected:

        std::vector<std::vector<float>> state;
        std::vector<std::vector<float>> q_values;
        std::vector<unsigned int>       action;
        std::vector<float>              reward;
        std::vector<float>              curiosity;
        std::vector<bool>               terminal;

        unsigned int buffer_size, current_ptr;
};


#endif
