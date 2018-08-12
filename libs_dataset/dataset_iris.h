#ifndef _DATASET_IRIS_H_
#define _DATASET_IRIS_H_

#include "dataset_interface.h"

class DatasetIRIS: public DatasetInterface
{
  public:
    DatasetIRIS(std::string file_name, float testing_ratio = 0.2);
    ~DatasetIRIS();

  private:
    struct sDatasetItem parse_line(std::string line);


    std::vector<float> parse_input(std::string str);
    std::vector<float> parse_output(std::string str);
};

#endif
