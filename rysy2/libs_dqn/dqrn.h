#ifndef _DQRN_H_
#define _DQRN_H_

#include <rnn.h>
#include <experience_replay_buffer.h>

class DQRN
{
    public:
        DQRN();
        DQRN(Shape state_shape, unsigned int actions_count, float gamma = 0.99, unsigned int replay_buffer_size = 4096, std::string network_config_file_name = "");
        DQRN(DQRN& other);
        DQRN(const DQRN& other);

        virtual ~DQRN();
        DQRN& operator= (DQRN& other);
        DQRN& operator= (const DQRN& other);

    protected:
        void copy(DQRN& other);
        void copy(const DQRN& other);

    public:
        void init(Shape state_shape, unsigned int actions_count, float gamma = 0.99, unsigned int replay_buffer_size = 4096, std::string network_config_file_name = "");
        void add_layer(std::string layer_type, Shape shape = {0, 0, 0});

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
        RNN *rnn;

        Shape state_shape;
        unsigned int actions_count;
        float gamma;

        ExperienceReplayBuffer experience_replay_buffer;

        std::vector<float> q_values;
        std::vector<float> v_state;

};

#endif
