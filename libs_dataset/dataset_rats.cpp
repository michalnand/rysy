#include "dataset_rats.h"
#include <math.h>


DatasetRats::DatasetRats(std::string data_file_name, float testing_ratio)
              :DatasetInterface()
{
  attributes_count = 43;

  channels = attributes_count;
  width = 1;
  height = 1;

  load_dataset(data_file_name, testing_ratio);
}

DatasetRats::~DatasetRats()
{

}


int DatasetRats::load_dataset(std::string data_file_name, float testing_ratio)
{
  FILE *f_data;
  f_data = fopen(data_file_name.c_str(),"r");

  if (f_data == nullptr)
  {
    printf("data file %s opening error\n", data_file_name.c_str());
    return -1;
  }

  std::vector<struct sDatasetItem> temporary;

  while (!feof(f_data))
  {
    struct sDatasetItem line;

    char lbl = 0;

    int res = fscanf(f_data,"%c ", &lbl);
    (void)res;

    line.output.resize(2);

    if (lbl == 'D')
    {
      line.output[0] = -1.0;
      line.output[1] =  1.0;
    }
    else
    {
      line.output[0] =  1.0;
      line.output[1] = -1.0;
    }

    for (unsigned int i = 0; i < attributes_count; i++)
    {
      float tmp = 0.0;
      res = fscanf(f_data,"%f ", &tmp);
      line.input.push_back(tmp);
    }

    temporary.push_back(line);
  }


  //normalise input attributes
  for (unsigned int j = 0; j < temporary[0].input.size(); j++)
  {
    float min = 1000000.0;
    float max = -min;

    for (unsigned int i = 0; i < temporary.size(); i++)
    {
      if (temporary[i].input[j] < min)
        min = temporary[i].input[j];

      if (temporary[i].input[j] > max)
        max = temporary[i].input[j];
    }

    if (max > min)
    {
      float k = (2.0)/(max - min);
      float q = 1.0 - k*max;

      for (unsigned int i = 0; i < temporary.size(); i++)
        temporary[i].input[j] = temporary[i].input[j]*k + q;
    }
    else
    {
      for (unsigned int i = 0; i < temporary.size(); i++)
        temporary[i].input[j] = 0.0;
    }
  }



  //random split into training and testing sets
  for (unsigned int j = 0; j < temporary.size(); j++)
  {
    float p = (rand()%10000)/10000.0;
    if (p < testing_ratio)
      testing.push_back(temporary[j]);
    else
      training.push_back(temporary[j]);
  }


  printf("Rats loading done, %u %u\n", get_training_size(), get_testing_size());
  return 0;
}
