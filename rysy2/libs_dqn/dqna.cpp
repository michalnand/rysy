#include <dqna.h>
#include <iostream>

DQNA::DQNA()
{
    features_network = nullptr;
    reconstruction_network = nullptr;
    q_network = nullptr;
}

DQNA::DQNA( Shape state_shape, unsigned int actions_count, float gamma, unsigned int replay_buffer_size,
            std::string features_network_config_file_name,
            std::string reconstruction_network_config_file_name,
            std::string q_network_config_file_name)
{
    features_network         = nullptr;
    reconstruction_network   = nullptr;
    q_network                = nullptr;

    init(   state_shape, actions_count, gamma, replay_buffer_size,
            features_network_config_file_name,
            reconstruction_network_config_file_name,
            q_network_config_file_name);
}

DQNA::DQNA(DQNA& other)
{
    copy(other);
}

DQNA::DQNA(const DQNA& other)
{
    copy(other);
}

DQNA::~DQNA()
{
    if (features_network != nullptr)
    {
        delete features_network;
        features_network = nullptr;
    }

    if (reconstruction_network != nullptr)
    {
        delete reconstruction_network;
        reconstruction_network = nullptr;
    }

    if (q_network != nullptr)
    {
        delete q_network;
        q_network = nullptr;
    }
}

DQNA& DQNA::operator= (DQNA& other)
{
    copy(other);
    return *this;
}

DQNA& DQNA::operator= (const DQNA& other)
{
    copy(other);
    return *this;
}

void DQNA::copy(DQNA& other)
{
    this->state_shape   = other.state_shape;
    this->actions_count = other.actions_count;
    this->gamma         = other.gamma;

    this->q_values      = other.q_values;

    this->experience_replay_buffer = other.experience_replay_buffer;
}

void DQNA::copy(const DQNA& other)
{
    this->state_shape   = other.state_shape;
    this->actions_count = other.actions_count;
    this->gamma         = other.gamma;

    this->q_values      = other.q_values;

    this->experience_replay_buffer = other.experience_replay_buffer;
}

void DQNA::init(    Shape state_shape, unsigned int actions_count, float gamma, unsigned int replay_buffer_size,
                    std::string features_network_config_file_name,
                    std::string reconstruction_network_config_file_name,
                    std::string q_network_config_file_name)
{
    this->state_shape   = state_shape;
    this->actions_count = actions_count;
    this->gamma         = gamma;

    this->q_values.resize(actions_count);

    this->experience_replay_buffer.init(replay_buffer_size, this->state_shape.size(), this->actions_count);

    Shape output_shape(1, 1, this->actions_count);

    if (features_network != nullptr)
    {
        delete features_network;
        features_network = nullptr;
    }

    if (reconstruction_network != nullptr)
    {
        delete reconstruction_network;
        reconstruction_network = nullptr;
    }

    if (q_network != nullptr)
    {
        delete q_network;
        q_network = nullptr;
    }


    features_network         = new CNN(features_network_config_file_name, this->state_shape);

    Shape features_shape = features_network->get_output_shape();

    reconstruction_network   = new CNN(reconstruction_network_config_file_name, features_shape, this->state_shape);
    q_network                = new CNN(q_network_config_file_name, features_shape, output_shape);

    features_output.resize(features_shape.size());


    t_state.init(state_shape);
    t_features_output.init(features_shape);
    t_reconstructed_state.init(state_shape);
    t_q_values.init(output_shape);

    t_reconstruction_error.init(state_shape);
    t_q_values_error.init(output_shape);
    t_features_error.init(features_shape);
}


std::vector<float>& DQNA::forward(std::vector<float> &state)
{
    features_network->forward(features_output, state);
    q_network->forward(q_values, features_output);

    return q_values;
}

std::vector<float>& DQNA::get_q_values()
{
    return q_values;
}

bool DQNA::add(std::vector<float>& state, std::vector<float>& q_values, unsigned int action, float reward, bool terminal)
{
    return experience_replay_buffer.add(state, q_values, action, reward, terminal);
}

bool DQNA::is_full()
{
    return experience_replay_buffer.is_full();
}

void DQNA::train()
{
    if (experience_replay_buffer.is_full() == false)
        return;

    experience_replay_buffer.compute(gamma);

    auto indices = make_indices(experience_replay_buffer.size());

    features_network->set_training_mode();
    reconstruction_network->set_training_mode();
    q_network->set_training_mode();


    for (unsigned int i = 0; i < indices.size(); i++)
    {
        unsigned int item_idx = indices[i];

        //obatin original state from buffer
        t_state.set_from_host(experience_replay_buffer.get_state()[item_idx]);

        //compute features
        features_network->forward(t_features_output, t_state);

        //compute reconstructed state
        reconstruction_network->forward(t_reconstructed_state, t_features_output);

        //compute q values
        q_network->forward(t_q_values, t_features_output);


        //compute reconstruction error as difference between target state and reconstruction output
        t_reconstruction_error = t_state;
        t_reconstruction_error.sub(t_reconstructed_state);

        //compute q_values error as difference between target and computed q_values
        t_q_values_error.set_from_host(experience_replay_buffer.get_q_values()[item_idx]);
        t_q_values_error.sub(t_q_values);

        //train reconstruction network
        reconstruction_network->train_from_error(t_reconstruction_error);

        //train q_network
        q_network->train_from_error(t_q_values_error);


        //train features network
        //use summed error from q_network and reconstruction_network
        /*
        t_features_error = q_network->get_error_back();
        t_features_error.add(reconstruction_network->get_error_back());
        features_network->train_from_error(t_features_error);
        */

        t_features_error = reconstruction_network->get_error_back();
        t_features_error.mul(0.1);
        t_features_error.add(q_network->get_error_back());
        features_network->train_from_error(t_features_error);
    }


    features_network->unset_training_mode();
    reconstruction_network->unset_training_mode();
    q_network->unset_training_mode();
}

void DQNA::print()
{
    std::cout << "features_network : \n";
    features_network->print();
    std::cout << "\n\n\n";

    std::cout << "reconstruction_network : \n";
    reconstruction_network->print();
    std::cout << "\n\n\n";

    std::cout << "q_network : \n";
    q_network->print();
    std::cout << "\n\n\n";
}

void DQNA::print_buffer()
{
    experience_replay_buffer.print();
}


void DQNA::save(std::string path)
{
    features_network->save(path + "features_network/");
    reconstruction_network->save(path + "reconstruction_network/");
    q_network->save(path + "q_network/");
}

void DQNA::load_weights(std::string file_name_prefix)
{
    features_network->load_weights(file_name_prefix + "features_network/");
    reconstruction_network->load_weights(file_name_prefix + "reconstruction_network/");
    q_network->load_weights(file_name_prefix + "q_network/");
}



std::vector<unsigned int> DQNA::make_indices(unsigned int count)
{
    std::vector<unsigned int> result(count);
    for (unsigned int i = 0; i < count; i++)
        result[i] = i;

    for (unsigned int i = 0; i < count; i++)
    {
        unsigned int idx_a = i;
        unsigned int idx_b = rand()%count;

        unsigned int tmp;

        tmp = result[idx_a];
        result[idx_a] = result[idx_b];
        result[idx_b] = tmp;
    }

    return result;
}
