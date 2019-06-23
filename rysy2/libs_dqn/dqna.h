#ifndef _DQNA_H_
#define _DQNA_H_

#include <cnn.h>
#include <experience_replay_buffer.h>

class DQNA
{
    public:
        DQNA();
        DQNA(   Shape state_shape, unsigned int actions_count, float gamma, unsigned int replay_buffer_size,
                std::string features_network_config_file_name,
                std::string reconstruction_network_config_file_name,
                std::string q_network_config_file_name);

        DQNA(DQNA& other);
        DQNA(const DQNA& other);

        virtual ~DQNA();
        DQNA& operator= (DQNA& other);
        DQNA& operator= (const DQNA& other);

    protected:
        void copy(DQNA& other);
        void copy(const DQNA& other);

    public:
        void init(  Shape state_shape, unsigned int actions_count, float gamma, unsigned int replay_buffer_size,
                    std::string features_network_config_file_name,
                    std::string reconstruction_network_config_file_name,
                    std::string q_network_config_file_name);

    public:
        std::vector<float>& forward(std::vector<float> &state);
        std::vector<float>& get_q_values();
        bool add(std::vector<float>& state, std::vector<float>& q_values, unsigned int action, float reward, bool terminal = false);

        bool is_full();
        void train();

    public:
        void print();
        void print_buffer();
        void save(std::string path);
        void load_weights(std::string file_name_prefix);

    protected:
        std::vector<unsigned int> make_indices(unsigned int count);

    protected:
        CNN *features_network;
        CNN *reconstruction_network;
        CNN *q_network;

        Shape state_shape;
        unsigned int actions_count;
        float gamma;

        ExperienceReplayBuffer experience_replay_buffer;

        std::vector<float> q_values;

        std::vector<float> features_output;

    protected:
        Tensor t_state;
        Tensor t_features_output;
        Tensor t_reconstructed_state;
        Tensor t_q_values;

        Tensor t_reconstruction_error;
        Tensor t_q_values_error;
        Tensor t_features_error;
};

#endif
