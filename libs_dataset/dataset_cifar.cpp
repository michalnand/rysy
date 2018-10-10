#include "dataset_cifar.h"
#include <limits>
#include <fstream>
#include <stdio.h>


DatasetCIFAR::DatasetCIFAR( std::string training_batch_file_name,
                            std::string testing_batch_file_name,
                            bool load_fine,
                            unsigned int padding)
{
  this->padding = padding;
  width     = 32 + 2*padding;
  height    = 32 + 2*padding;
  channels  = 3;

  load(training_batch_file_name, load_fine, false);
  load(testing_batch_file_name, load_fine, true);

  print();
}

DatasetCIFAR::~DatasetCIFAR()
{

}



void DatasetCIFAR::load(std::string file_name, bool load_fine, bool testing)
{
  unsigned int raw_width = 32;
  unsigned int raw_height = 32;

  unsigned int raw_input_size = raw_width*raw_height*channels;
  unsigned int input_size = width*height*channels;

  std::vector<unsigned char> raw;
  raw.resize(raw_input_size);

  FILE *f = fopen(file_name.c_str(), "r");

  unsigned int output_size = 20;
  if (load_fine)
    output_size = 100;

  training.resize(output_size);

  sDatasetItem item;

  item.input.resize(input_size);
  item.output.resize(output_size);

  for (unsigned int i = 0; i < item.input.size(); i++)
    item.input[i] = 0.0;

  while (!feof(f))
  {
    unsigned char raw_coarse_label  = 0;
    unsigned char raw_fine_label    = 0;

    int read_res;


    raw_coarse_label = fgetc(f);
    raw_fine_label   = fgetc(f);
    read_res = fread(&raw[0], raw_input_size, 1, f);
    (void)read_res;

    for (unsigned int ch = 0; ch < channels; ch++)
    {
      for (unsigned int y = 0; y < raw_height; y++)
        for (unsigned int x = 0; x < raw_width; x++)
        {
          float v = raw[(ch*raw_height + y)*raw_width + x]/256.0;
          item.input[(ch*height + y + padding)*width + x + padding] = v;
        }
    }

    for (unsigned int i = 0; i < output_size; i++)
      item.output[i] = 0.0;

    unsigned int label = raw_coarse_label;
    if (load_fine)
      label = raw_fine_label;

    if (label < output_size)
    {
      item.output[label] = 1.0;

      if (testing)
        add_testing(item);
      else
        add_training(item);
    }
  }

  fclose(f);
}
