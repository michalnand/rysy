#ifndef _ICM_H_
#define _ICM_H_

#include <cnn.h>
#include <experience_replay_buffer.h>


struct sICMResult
{
    float inverse_loss, forward_loss;
    unsigned int inverse_clasification_hit, inverse_clasification_miss;
    float inverse_classification_success;
};

class ICM
{
    public:
        ICM();
        ICM(Shape state_shape, unsigned int actions_count, std::string network_config_path);
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
        float forward(std::vector<float> &state_now, std::vector<float> &state_next, unsigned int action);

        float get_curiosity();

        sICMResult get_icm_result();

        void print();

    public:
        void save(std::string path);
        void load(std::string path);

    private:
        void train(std::vector<float> &state_now, std::vector<float> &state_next, unsigned int action);

        std::vector<float> action_vector(unsigned int action);
        float compute_curiosity();
        bool classification(Tensor &target, Tensor &output);

    private:
        Shape state_shape;
        unsigned int actions_count;

        Tensor t_state_now, t_state_next, t_action;
        Tensor t_features_now, t_features_next;

        Tensor t_inverse_input, t_inverse_output, t_inverse_error, t_inverse_error_back;
        Tensor t_forward_input, t_forward_output, t_forward_error, t_forward_error_back;


        Tensor t_features_now_error_inverse, t_features_next_error_inverse;
        Tensor t_features_now_error_forward, t_action_error;

        Tensor t_features_error_now, t_features_error_next;


        CNN *features_network, *inverse_network, *forward_network;

    private:
        float curiosity;

        sICMResult icm_result;

};


#endif
