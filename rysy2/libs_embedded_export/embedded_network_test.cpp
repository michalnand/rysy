#include <embedded_network_test.h>
#include <classification_compare.h>
#include <iostream>

EmbeddedNetworkTest::EmbeddedNetworkTest(DatasetInterface &dataset, EmbeddedNet &nn)
{
    this->dataset   = &dataset;
    this->nn        = &nn;
    this->io_scaling = 255.0;
}


EmbeddedNetworkTest::~EmbeddedNetworkTest()
{

}


float EmbeddedNetworkTest::process()
{
    unsigned int classes_count = dataset->get_output_shape().size();
    ClassificationCompare compare(classes_count);

    for (unsigned int item_idx = 0; item_idx < dataset->get_testing_count(); item_idx++)
    {
        auto nn_input = vfloat_to_nn_layer_t(dataset->get_testing_input(item_idx));
        nn->set_input(&nn_input[0]);

        nn->forward();

        auto target_output  = dataset->get_testing_output(item_idx);
        auto nn_output      = get_network_output(nn->get_output(), classes_count);

        compare.add(target_output, nn_output);
    }

    compare.compute();

    std::cout << compare.asString() << "\n";

    return compare.get_accuracy();
}


std::vector<nn_layer_t> EmbeddedNetworkTest::vfloat_to_nn_layer_t(std::vector<float> &v)
{
    std::vector<nn_layer_t> result(v.size());

    float max = v[0];
    float min = max;

    for (unsigned int i = 0; i < v.size(); i++)
    {
        if (v[i] > max)
            max = v[i];

        if (v[i] < min)
            min = v[i];
    }

    float k = 0.0;
    float q = 0.0;

    if (max > min)
    {
        k = (io_scaling - 0.0)*(max - min);
        q = 0.0 - k*min;
    }

    for (unsigned int i = 0; i < v.size(); i++)
        result[i] = k*v[i] + q;

    return result;
}

std::vector<float> EmbeddedNetworkTest::get_network_output(nn_layer_t* network_output, unsigned int size)
{
    std::vector<float> result(size);

    for (unsigned int i = 0; i < size; i++)
    {
        float tmp = network_output[i]/io_scaling;

        if (tmp > 1.0)
            tmp = 1.0;
        if (tmp < -1.0)
            tmp = -1.0;

        result[i] = tmp;
    }

    return result;
}
