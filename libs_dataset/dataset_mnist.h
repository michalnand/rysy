#ifndef _DATASET_MNIST_H_
#define _DATASET_MNIST_H_

#include "dataset_interface.h"

class DatasetMnist: public DatasetInterface
{
  private:
    unsigned int padding;

  public:
    DatasetMnist( std::string training_data_file_name, std::string training_labels_file_name,
                  std::string testing_data_file_name, std::string testing_labels_file_name,
                  unsigned int padding = 0);

    ~DatasetMnist();

  private:
    int load_dataset(std::string data_file_name, std::string labels_file_name, bool testing);

    unsigned int read_unsigned_int(FILE *f);
};

#endif
