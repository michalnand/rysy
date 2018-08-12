#ifndef _DATASET_TESTING_H_
#define _DATASET_TESTING_H_

#include "dataset_interface.h"

class DatasetTesting: public DatasetInterface
{
  private:
    float output_amplitude;
    std::vector<float> amplitude, frequency, phase;

  public:
    DatasetTesting(unsigned int dim = 1, float output_amplitude = 1.0);
    ~DatasetTesting();

  private:
    sDatasetItem create_item();
    float rnd();

};

#endif
