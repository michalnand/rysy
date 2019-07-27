#ifndef _DQN_CURIOSITY_H_
#define _DQN_CURIOSITY_H_

#include <cnn.h>
#include <icm.h>
#include <experience_replay_buffer.h>

class DQNCuriosity
{
    public:
        DQNCuriosity();
        DQNCuriosity(Shape state_shape, unsigned int actions_count, std::string config_path);
        DQNCuriosity(DQNCuriosity& other);
        DQNCuriosity(const DQNCuriosity& other);

        virtual ~DQNCuriosity();
        DQNCuriosity& operator= (DQNCuriosity& other);
        DQNCuriosity& operator= (const DQNCuriosity& other);

    protected:
        void copy(DQNCuriosity& other);
        void copy(const DQNCuriosity& other);

    public:
        void init(Shape state_shape, unsigned int actions_count, std::string config_path);

    public:
        std::vector<float>& forward(std::vector<float> &state);
        std::vector<float>& forward(float *state);
        std::vector<float>& get_q_values();
        bool add(std::vector<float>& state, std::vector<float>& q_values, unsigned int action, float reward, bool terminal = false);
        bool add(float *state, std::vector<float>& q_values, unsigned int action, float reward, bool terminal = false);

        bool is_full();
        void train();

    public:
        void print();
        void print_buffer();
        void save(std::string path);
        void load_weights(std::string file_name_prefix);

    protected:
        CNN *cnn;
        ICM *icm;

        Shape state_shape;
        unsigned int actions_count;
        float gamma;

        ExperienceReplayBuffer experience_replay_buffer;

        std::vector<float> q_values;
        std::vector<float> v_state;

};

#endif
