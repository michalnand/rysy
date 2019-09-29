#ifndef _DATASET_TIC_TAC_TOE_H_
#define _DATASET_TIC_TAC_TOE_H_

#include "dataset_interface.h"

class DatasetTicTacToe: public DatasetInterface
{
  public:
    DatasetTicTacToe(std::string data_file_name, float testing_ratio = 0.5, unsigned int padding = 0);

    ~DatasetTicTacToe();

  private:
    int load_dataset(std::string data_file_name, float testing_ratio, unsigned int padding);
    void rotate(unsigned int &y_, unsigned int &x_, unsigned int &y, unsigned int &x, unsigned int rotation);
};

#endif
