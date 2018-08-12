#ifndef _DATASET_WEIGHTS_TEMPERATURE_H_
#define _DATASET_WEIGHTS_TEMPERATURE_H_

#include "dataset_interface.h"

class DatasetWeightsTemperature: public DatasetInterface
{
  public:
    DatasetWeightsTemperature(std::string file_name);
    ~DatasetWeightsTemperature();

  private:
    int load_dataset(std::vector<sDatasetItem> *result, std::string data_file_name, float normalisation = 1.0);
};

#endif
