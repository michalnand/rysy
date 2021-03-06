#ifndef _DATASET_MNIST_RNN_H_
#define _DATASET_MNIST_RNN_H_

#include <dataset_interface.h>

class DatasetMnistRNN : public DatasetInterface
{
    public:
        DatasetMnistRNN(   std::string training_data_file_name, std::string training_labels_file_name,
                        std::string testing_data_file_name, std::string testing_labels_file_name);

        virtual ~DatasetMnistRNN();

    private:
        int load(  std::vector<std::vector<float>> &input, std::vector<std::vector<float>> &output,
                   std::string data_file_name, std::string labels_file_name);

        unsigned int read_unsigned_int(FILE *f);
};

#endif
