#include <cnn_visualisation.h>
#include <iostream>


CNNVisualisation::CNNVisualisation(CNN &nn)
{
    this->nn = &nn;

    for (unsigned int layer = 1; layer < this->nn->get_layer_output_size(); layer++)
    {
        auto shape   = this->nn->get_layer_output(layer).shape();

        if (correct_shape(shape))
            tensor_to_image.push_back(new TensorToImage(shape));
    }
}


CNNVisualisation::~CNNVisualisation()
{
    for (unsigned int layer = 0; layer < tensor_to_image.size(); layer++)
    {
        delete tensor_to_image[layer];
        tensor_to_image[layer] = nullptr;
    }
}

void CNNVisualisation::process()
{
    for (unsigned int layer = 0; layer < this->nn->get_layer_output_size(); layer++)
    {
        auto shape = this->nn->get_layer_output(layer+1).shape();

        if (correct_shape(shape))
        {
            std::string name = "layer " + std::to_string(layer);
            tensor_to_image[layer]->show(this->nn->get_layer_output(layer+1), name);
        }
    }
}

void CNNVisualisation::save(std::string file_name_prefix)
{
    for (unsigned int layer = 0; layer < this->nn->get_layer_output_size(); layer++)
    {
        auto shape = this->nn->get_layer_output(layer+1).shape();

        if (correct_shape(shape))
        {
            std::string file_name = file_name_prefix + "_" + std::to_string(layer) + ".png";
            tensor_to_image[layer]->save(this->nn->get_layer_output(layer+1), file_name);
        }
    }
}

bool CNNVisualisation::correct_shape(Shape shape)
{
    if (shape.w() < 4)
        return false;

    if (shape.h() < 4)
        return false;

    return true;
}
