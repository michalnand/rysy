#ifndef _DATASET_CIFAR_10_H_
#define _DATASET_CIFAR_10_H_

#include "dataset_interface.h"

class DatasetCIFAR10: public DatasetInterface
{
  private:
    unsigned int padding;
  public:
    DatasetCIFAR10( std::string training_batch_1_file_name,
                    std::string training_batch_2_file_name,
                    std::string training_batch_3_file_name,
                    std::string training_batch_4_file_name,
                    std::string training_batch_5_file_name,
                    std::string testing_batch_file_name,
                    unsigned int padding = 0);
    ~DatasetCIFAR10();

  private:

    void load(std::string file_name, bool testing);
};

#endif
