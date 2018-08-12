#ifndef _DATASET_INTERFACE_H_
#define _DATASET_INTERFACE_H_

#include <string>
#include <vector>
#include <stdio.h>


struct sDatasetItem
{
  std::vector<float> input;
  std::vector<float> output;
};

class DatasetInterface
{
  public:
    std::vector<sDatasetItem> unlabeled, training, testing;

  protected:
    unsigned int width;
    unsigned int height;
    unsigned int channels;

  public:
    DatasetInterface();
    virtual ~DatasetInterface();

    virtual unsigned int get_training_size();
    virtual unsigned int get_testing_size();
    virtual unsigned int get_unlabeled_size();

    unsigned int get_input_size();
    virtual unsigned int get_output_size();

    unsigned int get_width();
    unsigned int get_height();
    unsigned int get_channels();

    virtual sDatasetItem get_training(unsigned int idx);
    virtual sDatasetItem get_random_training();

    virtual sDatasetItem get_testing(unsigned int idx);
    virtual sDatasetItem get_random_testing();

    virtual sDatasetItem get_unlabeled(unsigned int idx);
    virtual sDatasetItem get_random_unlabeled();

    virtual sDatasetItem get_random_training(float noise);
    virtual sDatasetItem get_random_unlabeled(float noise);

    void print_training_item(unsigned int idx);
    void print_testing_item(unsigned int idx);


    unsigned int compare_biggest(unsigned int idx, char *output);

    void export_h_testing(std::string file_name, unsigned int count);

  protected:
    float rnd();
    unsigned int argmax(std::vector<float> &v);


};


#endif
