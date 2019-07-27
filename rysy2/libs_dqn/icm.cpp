#include <icm.h>


ICM::ICM()
{

}

ICM::ICM(Shape state_shape, unsigned int actions_count, unsigned int features_count, std::string network_config_dir)
{
    init(state_shape, actions_count, features_count, network_config_dir);
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
    state_shape     = other.state_shape;
    actions_count   = other.actions_count;
    features_count  = other.features_count;

    t_state_now     = other.t_state_now;
    t_state_next    = other.t_state_next;
    t_action        = other.t_action;

    t_features_now  = other.t_features_now;
    t_features_next = other.t_features_next;


    t_inverse_input         = other.t_inverse_input;
    t_inverse_output        = other.t_inverse_output;
    t_inverse_error         = other.t_inverse_error;
    t_inverse_error_back    = other.t_inverse_error_back;

    t_forward_input         = other.t_forward_input;
    t_forward_output        = other.t_forward_output;
    t_forward_error         = other.t_forward_error;
    t_forward_error_back    = other.t_forward_error_back;

    t_features_now_error_inverse    = other.t_features_now_error_inverse;
    t_features_next_error_inverse   = other.t_features_next_error_inverse;


    t_features_now_error_forward    = other.t_features_now_error_forward;
    t_action_error                  = other.t_action_error;

    t_features_error_now            = other.t_features_error_now;
    t_features_error_next           = other.t_features_error_next;
}

void ICM::copy(const ICM& other)
{
    state_shape     = other.state_shape;
    actions_count   = other.actions_count;
    features_count  = other.features_count;

    t_state_now     = other.t_state_now;
    t_state_next    = other.t_state_next;
    t_action        = other.t_action;

    t_features_now  = other.t_features_now;
    t_features_next = other.t_features_next;


    t_inverse_input         = other.t_inverse_input;
    t_inverse_output        = other.t_inverse_output;
    t_inverse_error         = other.t_inverse_error;
    t_inverse_error_back    = other.t_inverse_error_back;

    t_forward_input         = other.t_forward_input;
    t_forward_output        = other.t_forward_output;
    t_forward_error         = other.t_forward_error;
    t_forward_error_back    = other.t_forward_error_back;

    t_features_now_error_inverse    = other.t_features_now_error_inverse;
    t_features_next_error_inverse   = other.t_features_next_error_inverse;


    t_features_now_error_forward    = other.t_features_now_error_forward;
    t_action_error                  = other.t_action_error;

    t_features_error_now            = other.t_features_error_now;
    t_features_error_next           = other.t_features_error_next;
}

void ICM::init(Shape state_shape, unsigned int actions_count, unsigned int features_count, std::string network_config_path)
{
    this->state_shape       = state_shape;
    this->actions_count     = actions_count;
    this->features_count    = features_count;

    Shape features_output_shape(1, 1, this->features_count);
    Shape inverse_output_shape(1, 1, this->actions_count);
    Shape forward_input_shape(1, 1, this->features_count + this->actions_count);

    features_network = new CNN(network_config_path + "features_network.json", state_shape, features_output_shape);
    inverse_network  = new CNN(network_config_path + "inverse_network.json", features_output_shape, inverse_output_shape);
    forward_network  = new CNN(network_config_path + "forward_network.json", forward_input_shape, features_output_shape);


    t_state_now.init(state_shape);
    t_state_next.init(state_shape);
    t_action.init(1, 1, actions_count);

    t_features_now.init(features_output_shape);
    t_features_next.init(features_output_shape);

    t_inverse_input.init(1, 1, t_features_now.size() + t_features_next.size());
    t_inverse_output.init(1, 1, actions_count);
    t_inverse_error.init(features_output_shape);
    t_inverse_error_back.init(1, 1, t_features_now.size() + t_features_next.size());

    t_forward_input.init(forward_input_shape);
    t_forward_output.init(features_output_shape);
    t_forward_error.init(features_output_shape);
    t_forward_error_back.init(forward_input_shape);

    t_features_now_error_inverse.init(features_output_shape);
    t_features_next_error_inverse.init(features_output_shape);

    t_features_now_error_forward.init(features_output_shape);
    t_action_error.init(1, 1, actions_count);


    t_features_error_now.init(features_output_shape);
    t_features_error_next.init(features_output_shape);


    icm_result.inverse_loss = 0.0;
    icm_result.forward_loss = 0.0;
    icm_result.inverse_clasification_hit    = 0;
    icm_result.inverse_clasification_miss   = 0;
    icm_result.inverse_classification_success = 0.0;
}

void ICM::train(ExperienceReplayBuffer &replay_buffer)
{
    icm_result.inverse_loss = 0.0;
    icm_result.forward_loss = 0.0;
    icm_result.inverse_clasification_hit    = 0;
    icm_result.inverse_clasification_miss   = 0;
    icm_result.inverse_classification_success = 0.0;

    features_network->set_training_mode();
    inverse_network->set_training_mode();
    forward_network->set_training_mode();


    for (unsigned int i = 0; i < replay_buffer.size(); i++)
    {
        unsigned int idx = rand()%(replay_buffer.size()-1);
        train(replay_buffer.get_state()[idx], replay_buffer.get_state()[idx + 1], replay_buffer.get_action()[idx]);

        icm_result.inverse_loss+= t_inverse_error.norm_l2();
        icm_result.forward_loss+= t_forward_error.norm_l2();

        if (classification(t_action, t_inverse_output))
            icm_result.inverse_clasification_hit++;
        else
            icm_result.inverse_clasification_miss++;
    }

    icm_result.inverse_classification_success = (100.0*icm_result.inverse_clasification_hit)/(icm_result.inverse_clasification_hit + icm_result.inverse_clasification_miss);


    features_network->unset_training_mode();
    inverse_network->unset_training_mode();
    forward_network->unset_training_mode();
}



float ICM::forward(std::vector<float> &state_now, std::vector<float> &state_next, unsigned int action)
{
    t_state_now.set_from_host(state_now);
    t_state_next.set_from_host(state_next);
    auto v_action = action_vector(action);
    t_action.set_from_host(v_action);

    //compute state features
    features_network->forward(t_features_now, t_state_now);
    features_network->forward(t_features_next, t_state_next);

    t_forward_input.concatenate(t_features_now, t_action);
    forward_network->forward(t_forward_output, t_forward_input);

    //compute forward model error
    t_forward_error = t_features_next;
    t_forward_error.sub(t_forward_output);

    //use L2 norm to compute curiosity
    curiosity = t_forward_error.norm_l2()/t_forward_error.size();

    return curiosity;
}

float ICM::get_curiosity()
{
    return curiosity;
}

sICMResult ICM::get_icm_result()
{
    return icm_result;
}



void ICM::train(std::vector<float> &state_now, std::vector<float> &state_next, unsigned int action)
{
    t_state_now.set_from_host(state_now);
    t_state_next.set_from_host(state_next);
    auto v_action = action_vector(action);
    t_action.set_from_host(v_action);

    //compute state features
    features_network->forward(t_features_now, t_state_now);
    features_network->forward(t_features_next, t_state_next);

    //process inverse model output
    t_inverse_input.concatenate(t_features_now, t_features_next);
    inverse_network->forward(t_inverse_output, t_inverse_input);

    //process forward model output
    t_forward_input.concatenate(t_features_now, t_action);
    forward_network->forward(t_forward_output, t_forward_input);

    //error computation

    //compute inverse model error
    t_inverse_error = t_action;
    t_inverse_error.sub(t_inverse_output);


    //compute forward model error
    t_forward_error = t_features_next;
    t_forward_error.sub(t_forward_output);



    //training -> TODO this hell path
    inverse_network->train_from_error(t_inverse_error);
    t_inverse_error_back = inverse_network->get_error_back();

    forward_network->train_from_error(t_forward_error);
    t_forward_error_back = forward_network->get_error_back();


    t_inverse_error_back.split(t_features_now_error_inverse, t_features_next_error_inverse);
    t_forward_error_back.split(t_features_now_error_forward, t_action_error);


    t_features_error_now = t_features_now_error_inverse;
    t_features_error_now.add(t_features_now_error_forward);

    t_features_error_next = t_features_now_error_forward;


    features_network->forward(t_features_now, t_state_now);
    features_network->train_from_error(t_features_error_now);

    features_network->forward(t_features_next, t_state_next);
    features_network->train_from_error(t_features_error_next);
}


std::vector<float> ICM::action_vector(unsigned int action)
{
    std::vector<float> result(actions_count);

    for (unsigned int i = 0; i < actions_count; i++)
        result[i] = 0.0;
    result[action] = 1.0;

    return result;
}




bool ICM::classification(Tensor &target, Tensor &output)
{
    std::vector<float> v_target(target.size());
    target.set_to_host(v_target);

    std::vector<float> v_output(output.size());
    output.set_to_host(v_output);

    unsigned int max_target = 0;
    for (unsigned int i = 0; i < v_target.size(); i++)
        if (v_target[i] > v_target[max_target])
            max_target = i;


    unsigned int max_output = 0;
    for (unsigned int i = 0; i < v_output.size(); i++)
        if (v_output[i] > v_output[max_output])
                max_output = i;

    if (max_target == max_output)
        return true;

    return false;
}
