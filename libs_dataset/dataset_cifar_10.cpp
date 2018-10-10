#include "dataset_cifar_10.h"
#include <limits>
#include <fstream>
#include <stdio.h>


DatasetCIFAR10::DatasetCIFAR10( std::string training_batch_1_file_name,
                                std::string training_batch_2_file_name,
                                std::string training_batch_3_file_name,
                                std::string training_batch_4_file_name,
                                std::string training_batch_5_file_name,
                                std::string testing_batch_file_name,
                                unsigned int padding)
{
  this->padding = padding;
  width     = 32 + 2*padding;
  height    = 32 + 2*padding;
  channels  = 3;


  load(training_batch_1_file_name, false);
  load(training_batch_2_file_name, false);
  load(training_batch_3_file_name, false);
  load(training_batch_4_file_name, false);
  load(training_batch_5_file_name, false);

  load( testing_batch_file_name, true);

  print();
}

DatasetCIFAR10::~DatasetCIFAR10()
{

}



void DatasetCIFAR10::load(std::string file_name, bool testing)
{
  unsigned int raw_width    = 32;
  unsigned int raw_height   = 32;
  unsigned int output_size  = 10;

  unsigned int raw_input_size = raw_width*raw_height*channels;
  unsigned int input_size = width*height*channels;

  std::vector<unsigned char> raw;
  raw.resize(raw_input_size);

  FILE *f = fopen(file_name.c_str(), "r");

  sDatasetItem item;

  item.input.resize(input_size);
  item.output.resize(output_size);

  training.resize(output_size);

  for (unsigned int i = 0; i < item.input.size(); i++)
    item.input[i] = 0.0;

  for (unsigned int i = 0; i < item.output.size(); i++)
    item.output[i] = 0.0;

  while (!feof(f))
  {
    unsigned char label  = 0;

    int read_res;

    label = fgetc(f);
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
