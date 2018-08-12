#ifndef _DATASET_STL10_H_
#define _DATASET_STL10_H_

#include "dataset_interface.h"

class DatasetSTL10: public DatasetInterface
{
  private:


    unsigned int original_width;
    unsigned int original_height;

    unsigned int padding;


    std::string unlabeled_file_name;

    std::string train_file_name_x;
    std::string train_file_name_y;

    std::string testing_file_name_x;
    std::string testing_file_name_y;

  public:
    DatasetSTL10(unsigned int padding = 0);
    ~DatasetSTL10();

  private:
    void load_unlabeled(unsigned int count);
    void load_training();
    void load_testing();

    std::vector<float> load_input(FILE *f);
    std::vector<float> load_output(FILE *f);

    unsigned long int get_mem_availible();

};

#endif
