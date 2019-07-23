#ifndef _ICM_H_
#define _ICM_H_

#include <cnn.h>
#include <experience_replay_buffer.h>

class ICM
{
    public:
        ICM();
        ICM(Shape state_shape, unsigned int actions_count, std::string network_config_file_name);
        ICM(ICM& other);
        ICM(const ICM& other);

        virtual ~ICM();
        ICM& operator= (ICM& other);
        ICM& operator= (const ICM& other);

    protected:
        void copy(ICM& other);
        void copy(const ICM& other);

    public:
        void init(Shape state_shape, unsigned int actions_count, std::string network_config_path);

    public:
        void train(ExperienceReplayBuffer &replay_buffer);
        void train(std::vector<float> &state_now, std::vector<float> &state_next, unsigned int action);
        float forward(std::vector<float> &state_now, std::vector<float> &state_next, unsigned int action);

        float get_curiosity();

    private:
        std::vector<float> action_vector(unsigned int action);
        float compute_curiosity();


    private:
        Shape state_shape;
        unsigned int actions_count, features_count;

        Tensor t_state_now, t_features_now;
        Tensor t_state_next, t_features_next;
        Tensor t_features_now_next;
        Tensor t_inverse_target, t_inverse_output;

        Tensor t_action;
        Tensor t_forward_input;
        Tensor t_forward_output;


        CNN *features_network, *inverse_network, *forward_network;
        float curiosity;

};


#endif
