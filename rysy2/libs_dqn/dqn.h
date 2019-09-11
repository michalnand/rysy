#ifndef _DQN_H_
#define _DQN_H_

#include <cnn.h>
#include <experience_replay_buffer.h>

class DQN
{
    public:
        DQN();
        DQN(Shape state_shape, unsigned int actions_count, float gamma = 0.99, unsigned int replay_buffer_size = 4096, std::string network_config_file_name = "");
        DQN(DQN& other);
        DQN(const DQN& other);

        virtual ~DQN();
        DQN& operator= (DQN& other);
        DQN& operator= (const DQN& other);

    protected:
        void copy(DQN& other);
        void copy(const DQN& other);

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

    public:
        void kernel_visualisation(std::string path);
        void activity_visualisation(std::string path, std::vector<float> &state);
        void heatmap_visualisation(std::string path, std::vector<float> &state);



    protected:
        CNN *cnn;

        Shape state_shape;
        unsigned int actions_count;
        float gamma;

        ExperienceReplayBuffer experience_replay_buffer;

        std::vector<float> q_values;
        std::vector<float> v_state;

};

#endif
