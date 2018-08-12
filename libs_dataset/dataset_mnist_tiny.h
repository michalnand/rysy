#ifndef _DATASET_MNIST_TINY_H_
#define _DATASET_MNIST_TINY_H_

#include "dataset_interface.h"

class DatasetMnistTiny: public DatasetInterface
{
  private:
    int padding;
    
  public:
    DatasetMnistTiny(std::string training_data_file_name, std::string testing_data_file_name,
                     int padding = 0
                    );

    ~DatasetMnistTiny();

  private:
    int load_dataset(std::vector<struct sDatasetItem> *result, std::string data_file_name);
};

#endif
