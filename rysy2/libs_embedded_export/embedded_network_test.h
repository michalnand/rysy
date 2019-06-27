#ifndef _EMBEDDED_NETWORK_TEST_H_
#define _EMBEDDED_NETWORK_TEST_H_

#include <dataset_interface.h>
#include <EmbeddedNet.h>

class EmbeddedNetworkTest
{
    public:
        EmbeddedNetworkTest(DatasetInterface &dataset, EmbeddedNet &nn);
        virtual ~EmbeddedNetworkTest();


        float process();

    private:
        std::vector<nn_layer_t> vfloat_to_nn_layer_t(std::vector<float> &v);
        std::vector<float> get_network_output(nn_layer_t* network_output, unsigned int size);


    private:
        DatasetInterface *dataset;
        EmbeddedNet    *nn;
        float io_scaling;
};


#endif
