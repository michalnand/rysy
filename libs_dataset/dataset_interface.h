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
  protected:
    std::vector<std::vector<sDatasetItem>> training;
    std::vector<sDatasetItem> unlabeled, testing;

    unsigned int training_size, output_size;
    unsigned int width, height, channels;

  public:
    DatasetInterface();
    virtual ~DatasetInterface();

    void print();

  public:
    virtual sDatasetItem get_random_training();

    virtual sDatasetItem get_testing(unsigned int idx);
    virtual sDatasetItem get_random_testing();

    virtual sDatasetItem get_unlabeled(unsigned int idx);
    virtual sDatasetItem get_random_unlabeled();

  public:
    virtual unsigned int get_training_size()
    {
      return training_size;
    }

    virtual unsigned int get_testing_size()
    {
      return testing.size();
    }

    virtual unsigned int get_unlabeled_size()
    {
      return unlabeled.size();
    }

    virtual unsigned int get_input_size()
    {
      return width*height*channels;
    }

    virtual unsigned int get_output_size()
    {
      return output_size;
    }

    unsigned int get_width()
    {
      return width;
    }

    unsigned int get_height()
    {
      return height;
    }

    unsigned int get_channels()
    {
      return channels;
    }

  protected:
    void add_training(sDatasetItem &item);
    void add_testing(sDatasetItem &item);
    void add_unlabeled(sDatasetItem &item);

    unsigned int argmax(std::vector<float> &v);

  public:
    void print_testing_item(unsigned int idx);
    void export_h_testing(std::string file_name, unsigned int count);

};

#endif
