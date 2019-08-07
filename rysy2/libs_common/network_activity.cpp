#include <network_activity.h>


NetworkActivity::NetworkActivity(CNN &cnn)
{
    this->cnn = &cnn;

    for (unsigned int i = 0; i < this->cnn->get_layers_count(); i++)
        if (this->cnn->get_layer_weights_flag(i) == false)
        if (this->cnn->get_layer_output(i).shape().w() > 1)
        if (this->cnn->get_layer_output(i).shape().h() > 1)
            layer_idx.push_back(i);

    for (unsigned int i = 0; i < layer_idx.size(); i++)
    {
        Shape shape = this->cnn->get_layer_output(layer_idx[i]).shape();
        activity_maps.push_back(TensorToActivityMap(shape));
    }


    size_min = activity_maps[0].shape().w();
    size_max = size_min;

    for (unsigned int i = 0; i < activity_maps.size(); i++)
    {
        if (activity_maps[i].shape().w() < size_min)
            size_min = activity_maps[i].shape().w();

        if (activity_maps[i].shape().h() < size_min)
            size_min = activity_maps[i].shape().h();

        if (activity_maps[i].shape().w() > size_max)
            size_max = activity_maps[i].shape().w();

        if (activity_maps[i].shape().h() > size_max)
            size_max = activity_maps[i].shape().h();
    }
}

NetworkActivity::~NetworkActivity()
{

}

void NetworkActivity::add()
{
    for (unsigned int i = 0; i < layer_idx.size(); i++)
    {
        activity_maps[i].add(this->cnn->get_layer_output(layer_idx[i]));
    }
}

void NetworkActivity::save(std::string output_path)
{
    for (unsigned int i = 0; i < layer_idx.size(); i++)
    {
        unsigned int size   = this->cnn->get_layer_output(layer_idx[i]).shape().w();
        unsigned int ratio  = 4*(size_max/size);

        std::string name = output_path + std::to_string(layer_idx[i]);
        activity_maps[i].save(name, ratio);
    }

    for (unsigned int i = 0; i < activity_maps.size(); i++)
        activity_maps[i].clear();
}
