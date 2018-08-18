#ifndef _DATASET_LANDSAT_H_
#define _DATASET_LANDSAT_H_

#include "dataset_interface.h"

class DatasetLANDSAT: public DatasetInterface
{
  public:
    DatasetLANDSAT(std::string training_data_file_name, std::string testing_data_file_name, unsigned int padding);
    ~DatasetLANDSAT();


  private:
    int load_dataset(std::string data_file_name, unsigned int padding, bool testing);
};

#endif
