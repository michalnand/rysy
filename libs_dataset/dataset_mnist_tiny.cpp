#include "dataset_mnist_tiny.h"
#include <math.h>


DatasetMnistTiny::DatasetMnistTiny( std::string training_data_file_name,
                                    std::string testing_data_file_name,
                                    int padding)
              :DatasetInterface()
{
  this->padding = padding;
  channels  = 1;
  width     = 9 + 2*padding;
  height    = 9 + 2*padding;

  load_dataset(training_data_file_name, false);
  load_dataset(testing_data_file_name, true);
  
  print();
}

DatasetMnistTiny::~DatasetMnistTiny()
{

}


int DatasetMnistTiny::load_dataset(std::string data_file_name, bool testing)
{
  unsigned int original_width  = 9;
  unsigned int original_height = 9;

  std::vector<float> raw;
  raw.resize(original_width*original_height);

  struct sDatasetItem item;

  item.input.resize(height*width);
  item.output.resize(10);


  FILE *f_data;
  f_data = fopen(data_file_name.c_str(),"r");

  if (f_data == nullptr)
  {
    printf("data file %s opening error\n", data_file_name.c_str());
    return -1;
  }

  int res;

  while (!feof(f_data))
  {
    float tmp = 0.0;

    for (unsigned int j = 0; j < (original_width*original_height); j++)
    {
      res = fscanf(f_data,"%f ", &tmp);
      raw[j] = tmp;
    }

    for (unsigned int y = 0; y < original_height; y++)
      for (unsigned int x = 0; x < original_width; x++)
      {
        unsigned int input_idx  = y*original_height + x;
        unsigned int output_idx = (y + padding)*height + x + padding;

        float v = raw[input_idx];
        if (v < 0.0)
          v = 0.0;
        item.input[output_idx] = v;
      }

    for (unsigned int j = 0; j < 10; j++)
    {
      res = fscanf(f_data,"%f ", &tmp);

      if (tmp > 0)
        item.output[j] = 1.0;
      else
        item.output[j] = 0.0;
    }

    if (testing)
      add_testing(item);
    else
      add_training(item);
  }

  (void)res;
  fclose(f_data);

  return 0;
}
