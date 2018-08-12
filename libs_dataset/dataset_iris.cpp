#include "dataset_iris.h"

#include <algorithm>
#include <sstream>

DatasetIRIS::DatasetIRIS(std::string file_name, float testing_ratio)
               :DatasetInterface()
{
  channels  = 4;
  width     = 1;
  height    = 1;

  FILE *f = fopen(file_name.c_str(),"r");

  if (f == nullptr)
  {
    printf("file %s opening error\n", file_name.c_str());
    return;
  }

  std::vector<struct sDatasetItem> temporary;

  while (!feof(f))
  {
    char line[1024];
    int res = fscanf(f, "%s\n", line);
    (void)res;

    std::string tmp = line;
    struct sDatasetItem item = parse_line(tmp);

    temporary.push_back(item);
  }


  fclose(f);



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


    float k = (2.0)/(max - min);
    float q = 1.0 - k*max;

    for (unsigned int i = 0; i < temporary.size(); i++)
      temporary[i].input[j] = temporary[i].input[j]*k + q;
  }




  for (unsigned int i = 0; i < temporary.size(); i++)
  {
    float rnd = (rand()%10000)/10000.0;
    if (rnd < testing_ratio)
      testing.push_back(temporary[i]);
    else
      training.push_back(temporary[i]);
  }

  /*
  for (unsigned int j = 0; j < training.size(); j++)
    print_training_item(j);

  for (unsigned int j = 0; j < testing.size(); j++)
    print_testing_item(j);
    */
//  printf("DatasetIRIS loading done %u %u %u %u\n", get_training_size(), get_testing_size(), get_input_size(), get_output_size());
}


DatasetIRIS::~DatasetIRIS()
{

}


struct sDatasetItem DatasetIRIS::parse_line(std::string line)
{
  struct sDatasetItem result;

  result.input = parse_input(line);
  result.output = parse_output(line);

  return result;
}

std::vector<float> DatasetIRIS::parse_input(std::string str)
{
  std::vector<float> result;

  std::replace( str.begin(), str.end(), ',', ' ');

  std::stringstream stream(str);


//  while(stream)
  for (unsigned int i = 0; i < 4; i++)
  {
    float value;
    stream >> value;

    result.push_back(value);
  }

  return result;
}

std::vector<float> DatasetIRIS::parse_output(std::string str)
{
  std::vector<float> result;

  for (unsigned int i = 0; i < 3; i++)
    result.push_back(-1.0);

  if (std::string::npos != str.find("Iris-setosa"))
    result[0] = 1.0;
  else
  if (std::string::npos != str.find("Iris-versicolor"))
    result[1] = 1.0;
  else
  if (std::string::npos != str.find("Iris-virginica"))
    result[2] = 1.0;

  return result;
}
