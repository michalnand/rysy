#ifndef _DATASET_AUTOENCODER_H_
#define _DATASET_AUTOENCODER_H_

#include <dataset_interface.h>

class DatasetAutoencoder : public DatasetInterface
{
    public:
        DatasetAutoencoder(DatasetInterface &dataset, bool clear_original_dataset = true);
        virtual ~DatasetAutoencoder();

    protected:
        virtual void processing_init();

        virtual std::vector<float> process_training_input(std::vector<float> &v);
        virtual std::vector<float> process_training_output(std::vector<float> &v);
        virtual std::vector<float> process_testing_input(std::vector<float> &v);
        virtual std::vector<float> process_testing_output(std::vector<float> &v);

    protected:
        float rnd(float min = -1.0, float max = 1.0);
};

#endif
