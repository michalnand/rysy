#include "dataset_landsat.h"


DatasetLANDSAT::DatasetLANDSAT(std::string training_data_file_name, std::string testing_data_file_name, unsigned int padding)
              :DatasetInterface()
{

  channels  = 4;
  width     = 3 + 2*padding;
  height    = 3 + 2*padding;

  output_size = 8;
  training.resize(output_size);

  load_dataset(training_data_file_name, padding, false);
  load_dataset(testing_data_file_name, padding, true);

  print();
}

DatasetLANDSAT::~DatasetLANDSAT()
{

}


int DatasetLANDSAT::load_dataset(std::string data_file_name, unsigned int padding, bool testing)
{
  FILE *f_data;
  f_data = fopen(data_file_name.c_str(),"r");

  if (f_data == nullptr)
  {
    printf("data file %s opening error\n", data_file_name.c_str());
    return -1;
  }

  int res;

  unsigned int orignal_width = 3;
  unsigned int orignal_height = 3;
  unsigned int orignal_channels = 4;

  while (!feof(f_data))
  {
    struct sDatasetItem item;

    int tmp = 0;

    std::vector<float> raw;

    for (unsigned int j = 0; j < orignal_width*orignal_height*orignal_channels; j++)
    {
      res = fscanf(f_data,"%i ", &tmp);
      raw.push_back(tmp/256.0);
    }

    item.input.resize(height*width*channels);

    for (unsigned int ch = 0; ch < orignal_channels; ch++)
      for (unsigned int j = 0; j < orignal_height; j++)
        for (unsigned int i = 0; i < orignal_width; i++)
        {
          unsigned int input_idx = (j*orignal_width + i)*orignal_channels + ch;
          unsigned int output_idx = (ch*height + j + padding)*width + i + padding;
          item.input[output_idx] = raw[input_idx];
        }

    unsigned int label = 0;
    for (unsigned int i = 0; i < 7; i++)
      item.output.push_back(-1.0);

    res = fscanf(f_data,"%u\n", &label);

    label-= 1;

    if (label < item.output.size())
      item.output[label] = 1.0;

    if (testing)
      add_testing(item);
    else
      add_training(item);
  }

  (void)res;
  fclose(f_data);

  return 0;
}
