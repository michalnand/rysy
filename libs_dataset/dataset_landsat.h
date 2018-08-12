#ifndef _DATASET_LANDSAT_H_
#define _DATASET_LANDSAT_H_

#include "dataset_interface.h"

class DatasetLANDSAT: public DatasetInterface
{
  public:
    DatasetLANDSAT(std::string training_data_file_name, std::string testing_data_file_name, unsigned int padding);

    ~DatasetLANDSAT();

    void print(unsigned int idx);

  private:
    int load_dataset(std::vector<sDatasetItem> *result, std::string data_file_name, unsigned int padding);
};

#endif
