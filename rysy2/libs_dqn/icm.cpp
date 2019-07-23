#include <icm.h>


ICM::ICM()
{

}

ICM::ICM(Shape state_shape, unsigned int actions_count, std::string network_config_dir)
{
    init(state_shape, actions_count, network_config_dir);
}

ICM::ICM(ICM& other)
{
    copy(other);
}

ICM::ICM(const ICM& other)
{
    copy(other);
}

ICM::~ICM()
{
    delete features_network;
    delete inverse_network;
    delete forward_network;
}



ICM& ICM::operator= (ICM& other)
{
    copy(other);
    return *this;
}

ICM& ICM::operator= (const ICM& other)
{
    copy(other);
    return *this;
}

void ICM::copy(ICM& other)
{
    state_shape   = other.state_shape;
    actions_count = other.actions_count;
}

void ICM::copy(const ICM& other)
{
    state_shape   = other.state_shape;
    actions_count = other.actions_count;
}

void ICM::init(Shape state_shape, unsigned int actions_count, std::string network_config_path)
{
    this->state_shape   = state_shape;
    this->actions_count = actions_count;
    this->features_count = 256;

    Shape features_output_shape(1, 1, this->features_count);
    Shape inverse_output_shape(1, 1, this->actions_count);
    Shape forward_input_shape(1, 1, this->features_count + this->actions_count);
 
    features_network = new CNN(network_config_path + "features_network.json", state_shape, features_output_shape);
    inverse_network  = new CNN(network_config_path + "inverse_network.json", features_output_shape, inverse_output_shape);
    forward_network  = new CNN(network_config_path + "forward_network.json", forward_input_shape, features_output_shape);

    t_state_now.init(state_shape);
    t_features_now.init(features_output_shape);

    t_state_next.init(state_shape);
    t_features_next.init(features_output_shape);

    t_features_now_next.init(forward_input_shape);

    t_inverse_target.init(inverse_output_shape);
    t_inverse_output.init(inverse_output_shape);

    t_action.init(1, 1, actions_count);
    t_forward_input.init(forward_input_shape);
    t_forward_output.init(forward_input_shape);
}

void ICM::train(ExperienceReplayBuffer &replay_buffer)
{
    for (unsigned int i = 0; i < replay_buffer.size(); i++)
    {
        unsigned int idx = rand()%(replay_buffer.size()-1);
        train(replay_buffer.get_state()[idx], replay_buffer.get_state()[idx + 1], replay_buffer.get_action()[idx]);
    }
}

void ICM::train(std::vector<float> &state_now, std::vector<float> &state_next, unsigned int action)
{
    //TODO
}



float ICM::forward(std::vector<float> &state_now, std::vector<float> &state_next, unsigned int action)
{
    t_state_now.set_from_host(state_now);
    t_state_next.set_from_host(state_next);

    features_network->forward(t_features_now, t_state_now);
    features_network->forward(t_features_next, t_state_next);

    auto v_action = action_vector(action);
    t_action.set_from_host(v_action);

    t_forward_input.concatenate(t_features_now, t_action);
    forward_network->forward(t_forward_output, t_forward_input);

    curiosity = compute_curiosity();
    return curiosity;
}

float ICM::get_curiosity()
{
    return curiosity;
}

std::vector<float> ICM::action_vector(unsigned int action)
{
    std::vector<float> result(actions_count);

    for (unsigned int i = 0; i < actions_count; i++)
        result[i] = 0.0;
    result[action] = 1.0;

    return result;
}

float ICM::compute_curiosity()
{
    std::vector<float> v_features_target(features_count);
    std::vector<float> v_features_output(features_count);

    t_features_next.set_to_host(v_features_target);
    t_forward_output.set_to_host(v_features_output);

    float sum = 0.0;
    for (unsigned int i = 0; i < v_features_target.size(); i++)
    {
        float tmp = v_features_target[i] - v_features_output[i];
        sum+= tmp*tmp;
    }

    return sum/v_features_target.size();
}
