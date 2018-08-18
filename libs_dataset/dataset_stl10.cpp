#include "dataset_stl10.h"
#include <limits>
#include <fstream>

DatasetSTL10::DatasetSTL10(unsigned int padding)
             :DatasetInterface()
{
  this->padding = padding;

  original_width  = 96;
  original_height = 96;

  width     = original_width  + 2*padding;
  height    = original_height + 2*padding;
  channels  = 3;

  unlabeled_file_name = "/home/michal/dataset/stl10_binary/unlabeled_X.bin";

  train_file_name_x = "/home/michal/dataset/stl10_binary/train_X.bin";
  train_file_name_y = "/home/michal/dataset/stl10_binary/train_y.bin";

  testing_file_name_x = "/home/michal/dataset/stl10_binary/test_X.bin";
  testing_file_name_y = "/home/michal/dataset/stl10_binary/test_y.bin";

  load_training();
  load_testing();

/*
  unsigned int required_size = 4*width*height*channels;
  unsigned int max_mem   = get_mem_availible()/1000;  //max_mem in MB
  if (max_mem < 4096)
   max_mem = 4096;
  unsigned int max_count = max_mem/(required_size*0.000001);

  load_unlabeled(max_count);
*/
  print();
}

DatasetSTL10::~DatasetSTL10()
{

}




void DatasetSTL10::load_unlabeled(unsigned int count)
{
  FILE *f;

  f = fopen(unlabeled_file_name.c_str(), "r");

  for (unsigned int i = 0; i < count; i++)
  {
    sDatasetItem item;
    item.input = load_input(f);

    add_unlabeled(item);

    if ((i%1000) == 0)
      printf("loading unlabeled dataset %u/%u %6.2f %%\n", i, count, i*100.0/count);
  }

  fclose(f);
}

void DatasetSTL10::load_training()
{
  FILE *fx, *fy;

  fx = fopen(train_file_name_x.c_str(), "r");
  fy = fopen(train_file_name_y.c_str(), "r");

  unsigned int count = 0;
  while (!feof(fx))
  {
    sDatasetItem item;
    item.input = load_input(fx);
    item.output = load_output(fy);

    add_training(item);

    count++;
    if (count >= 5000)
      break;
  }

  fclose(fx);
  fclose(fy);
}

void DatasetSTL10::load_testing()
{
  FILE *fx, *fy;

  fx = fopen(testing_file_name_x.c_str(), "r");
  fy = fopen(testing_file_name_y.c_str(), "r");

  unsigned int count = 0;
  while (!feof(fx))
  {
    sDatasetItem item;
    item.input = load_input(fx);
    item.output = load_output(fy);

    add_testing(item);


    count++;
    if (count >= 8000)
      break;
  }

  fclose(fx);
  fclose(fy);
}

std::vector<float> DatasetSTL10::load_input(FILE *f)
{
  unsigned int original_size   = 3*original_width*original_height;
  unsigned int size            = width*height*channels;

  std::vector<unsigned char> raw_data;
  std::vector<float> result;

  raw_data.resize(original_size);
  result.resize(size);


  int res = fread(&raw_data[0], sizeof(unsigned char), original_size, f);
  (void)res;

  unsigned int original_width  = 96;
  unsigned int original_height = 96;

  for (unsigned int channel = 0; channel < channels; channel++)
  for (unsigned int y = 0; y < original_height; y++)
  for (unsigned int x = 0; x < original_width; x++)
  {
    unsigned int input_idx  = (channel*original_height + x)*original_width + y;
    unsigned int output_idx = (channel*height + y + padding)*width + x + padding;

    result[output_idx] = raw_data[input_idx]/256.0;
  }

  return result;
}

std::vector<float> DatasetSTL10::load_output(FILE *f)
{
  std::vector<float> result;
  result.resize(10);

  unsigned char readed = 0;

  int res = fread(&readed, sizeof(unsigned char), 1, f);
  (void)res;

  for (unsigned int i = 0; i < result.size(); i++)
    result[i] = -1.0;

  readed = readed-1;
  if (readed < result.size())
    result[readed] = 1.0;

  return result;
}

unsigned long int DatasetSTL10::get_mem_availible()
{
    std::string token;
    std::ifstream file("/proc/meminfo");
    while(file >> token)
    {
        if(token == "MemAvailable:")
        {
            unsigned long mem;
            if(file >> mem)
            {
                return mem;
            } else
            {
              return 0;
            }
        }

        // ignore rest of the line
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    return 0; // nothing found
}
