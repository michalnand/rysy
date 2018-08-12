#ifndef _DATASET_CIFAR_H_
#define _DATASET_CIFAR_H_

#include "dataset_interface.h"

class DatasetCIFAR: public DatasetInterface
{
  private:
    unsigned int padding;
  public:
    DatasetCIFAR(std::string training_batch_file_name,
                   std::string testing_batch_file_name,
                   bool load_fine = false,
                   unsigned int padding = 0);
    ~DatasetCIFAR();

  private:

    void load(std::vector<sDatasetItem> &result, std::string file_name, bool load_fine);
};

#endif
