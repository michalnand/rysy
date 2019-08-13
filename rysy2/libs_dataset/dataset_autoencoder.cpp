#include <dataset_autoencoder.h>
#include <iostream>


DatasetAutoencoder::DatasetAutoencoder(DatasetInterface &dataset, bool clear_original_dataset)
                   :DatasetInterface()    
{
    std::cout << "DatasetAutoencoder initialising\n";

    input_shape     = dataset.get_input_shape();
    output_shape    = dataset.get_input_shape();

    training_input.resize(dataset.get_training_count());
    training_output.resize(dataset.get_training_count());
    testing_input.resize(dataset.get_training_count());
    testing_output.resize(dataset.get_testing_count());

    std::cout << "DatasetAutoencoder moving from dataset\n";

    for (unsigned int i = 0; i < dataset.get_training_count(); i++)
    {
        dataset.set_training_idx(i);
        training_input[i] = dataset.get_training_input();
    }


    for (unsigned int i = 0; i < dataset.get_testing_count(); i++)
    {
        testing_input[i] = dataset.get_testing_input(i);
    }

    if (clear_original_dataset)
    {
        std::cout << "clearning original dataset\n";
        dataset.clear();
    }

    std::cout << "DatasetAutoencoder processing\n";

    processing_init();

    for (unsigned int i = 0; i < get_training_count(); i++)
    {
        auto tmp = training_input[i];
        training_input[i]  = process_training_input(tmp);
        training_output[i] = process_training_output(tmp);
    }

    for (unsigned int i = 0; i < get_testing_count(); i++)
    {
        auto tmp = testing_input[i];
        testing_input[i]  = process_testing_input(tmp);
        testing_output[i] = process_testing_output(tmp);
    }

    std::cout << "DatasetAutoencoder init done\n";
}


DatasetAutoencoder::~DatasetAutoencoder()
{

}

void DatasetAutoencoder::processing_init()
{

}

std::vector<float> DatasetAutoencoder::process_training_input(std::vector<float> &v)
{
    float noise = 0.5;

    std::vector<float> result(v.size());

    for (unsigned int i = 0; i < result.size(); i++)
        result[i] = (1.0 - noise)*v[i] + noise*rnd(-1.0, 1.0);

    return result;
}

std::vector<float> DatasetAutoencoder::process_training_output(std::vector<float> &v)
{
    return v;
}

std::vector<float> DatasetAutoencoder::process_testing_input(std::vector<float> &v)
{
    float noise = 0.5;

    std::vector<float> result(v.size());

    for (unsigned int i = 0; i < result.size(); i++)
        result[i] = (1.0 - noise)*v[i] + noise*rnd(-1.0, 1.0);

    return result;
}

std::vector<float> DatasetAutoencoder::process_testing_output(std::vector<float> &v)
{
    return v;
}

float DatasetAutoencoder::rnd(float min, float max)
{
    float v = (rand()%10000000)/10000000.0;

    return (max - min)*v + min;
}
