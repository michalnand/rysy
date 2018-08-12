#include "dataset_weights_temperature.h"


DatasetWeightsTemperature::DatasetWeightsTemperature(std::string file_name)
              :DatasetInterface()
{
  channels = 2;
  width    = 1;
  height   = 1;

  load_dataset(&training, file_name, 100.0);
  load_dataset(&testing, file_name, 100.0);
}

DatasetWeightsTemperature::~DatasetWeightsTemperature()
{

}


int DatasetWeightsTemperature::load_dataset(std::vector<struct sDatasetItem> *result, std::string data_file_name, float normalisation)
{
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
    struct sDatasetItem item;

    std::vector<float> raw;

    for (unsigned int j = 0; j < height*width*channels; j++)
    {
      float tmp = 0;
      res = fscanf(f_data,"%f ", &tmp);
      tmp/= normalisation;
      raw.push_back(tmp);
    }

    item.input = raw;

    float raw_output = 0;
    res = fscanf(f_data,"%f\n", &raw_output);
    raw_output/= normalisation;
 
    item.output.push_back(raw_output);

    result->push_back(item);

/*
    for (unsigned int i = 0; i < item.input.size(); i++)
      printf("%6.3f ", item.input[i]);

    printf(" : ");

    for (unsigned int i = 0; i < item.output.size(); i++)
      printf("%6.3f ", item.output[i]);
    printf("\n");
  */
  }

  (void)res;
  fclose(f_data);

  printf("loading done count %u\n", (unsigned int)result->size());


  return 0;
}
