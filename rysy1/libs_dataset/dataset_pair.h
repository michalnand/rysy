#ifndef _DATASET_PAIR_H_
#define _DATASET_PAIR_H_

#include "dataset_interface.h"

class DatasetPair: public DatasetInterface
{
  protected:
    unsigned int testing_size;

  public:
    DatasetPair(DatasetInterface &dataset, int training_size_ = -1, int testing_size_ = -1);

    ~DatasetPair();

  private:
    void create(DatasetInterface &dataset, unsigned int count, bool set_testing);

};

#endif
