#include "dataset_landsat.h"


DatasetLANDSAT::DatasetLANDSAT(std::string training_data_file_name, std::string testing_data_file_name, unsigned int padding)
              :DatasetInterface()
{

  channels  = 4;
  width     = 3 + 2*padding;
  height    = 3 + 2*padding;

  load_dataset(&training, training_data_file_name, padding);
  load_dataset(&testing, testing_data_file_name, padding);
}

DatasetLANDSAT::~DatasetLANDSAT()
{

}


int DatasetLANDSAT::load_dataset(std::vector<struct sDatasetItem> *result, std::string data_file_name, unsigned int padding)
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

  printf("LANDSAT loading done count %u\n", (unsigned int)result->size());


  return 0;
}
