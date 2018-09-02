#ifndef _DATASET_LINE_H_
#define _DATASET_LINE_H_

#include "dataset_interface.h"

#include <json_config.h>

class DatasetLine: public DatasetInterface
{
  private:
    unsigned int classes_count;

  public:
    DatasetLine();
    ~DatasetLine();

  private:
    void create(unsigned int count, bool testing);
    sDatasetItem create_item();

    float rnd(float min = -1.0, float max = 1.0);
    void set_input(std::vector<float> &input, int x, int y, float value);

};

#endif
