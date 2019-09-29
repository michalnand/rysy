#ifndef _DATASET_BINARY_H_
#define _DATASET_BINARY_H_

#include "dataset_interface.h"

class DatasetBinary: public DatasetInterface
{
  public:
    DatasetBinary(  std::string training_file_name,
                    std::string testing_file_name);

    virtual ~DatasetBinary();

};

#endif
