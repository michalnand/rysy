#ifndef _DATASET_RATS_H_
#define _DATASET_RATS_H_

#include "dataset_interface.h"

class DatasetRats: public DatasetInterface
{
  private:
    unsigned int attributes_count = 43;

  public:
    DatasetRats(std::string data_file_name, float testing_ratio = 0.5);

    ~DatasetRats();

  private:
    int load_dataset(std::string data_file_name, float testing_ratio);
};

#endif
