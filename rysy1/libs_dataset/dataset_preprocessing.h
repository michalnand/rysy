#ifndef _DATASET_PREPROCESSING_H_
#define _DATASET_PREPROCESSING_H_

#include "dataset_interface.h"
#include <preprocessing.h>

class DatasetPreprocessing: public DatasetInterface
{
  private:
    Preprocessing preprocessing;

  public:
    DatasetPreprocessing(DatasetInterface &dataset, std::string config_file_name);
    ~DatasetPreprocessing();
};

#endif
