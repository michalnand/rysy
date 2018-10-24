#include "dataset_interface.h"

#include <iostream>
#include <log.h>
#include <json_config.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

DatasetInterface::DatasetInterface()
{
  srand(time(NULL));

  width     = 0;
  height    = 0;
  channels  = 0;

  training_size = 0;
  output_size = 0;

  training.resize(1);
}

DatasetInterface::~DatasetInterface()
{

}

void DatasetInterface::print()
{
  std::cout << "training size   " << get_training_size() << "\n";
  std::cout << "testing size    " << get_testing_size() << "\n";
  std::cout << "unlabeled size  " << get_unlabeled_size() << "\n";
  std::cout << "\n";
  std::cout << "classes count   " << get_output_size() << "\n";;
  std::cout << "geometry        " << get_width() << " " << get_height() << " " << get_channels() << "\n";
  std::cout << "\n";
}

sDatasetItem DatasetInterface::get_random_training()
{
  unsigned int class_id;

  do
  {
    class_id = rand()%training.size();
  }
  while (training[class_id].size() == 0);

  unsigned int item_id  = rand()%training[class_id].size();

  return training[class_id][item_id];
}

sDatasetItem DatasetInterface::get_training(unsigned int class_idx, unsigned int idx)
{
  return training[class_idx][idx];
}

unsigned int DatasetInterface::get_class_items_count(unsigned int class_idx)
{
  return training[class_idx].size();
}

sDatasetItem DatasetInterface::get_testing(unsigned int idx)
{
  return testing[idx];
}

sDatasetItem DatasetInterface::get_random_testing()
{
  return get_testing(rand()%get_testing_size());
}

sDatasetItem DatasetInterface::get_unlabeled(unsigned int idx)
{
  return unlabeled[idx];
}

sDatasetItem DatasetInterface::get_random_unlabeled()
{
  return get_unlabeled(rand()%get_unlabeled_size());
}





void DatasetInterface::add_training(sDatasetItem &item)
{
  unsigned int class_id = argmax(item.output);

  if (class_id >= training.size())
  {
    training.resize(class_id);
  }

  training[class_id].push_back(item);

  output_size   = item.output.size();
  training_size = 0;

  for (unsigned int j = 0; j < training.size(); j++)
    training_size+= training[j].size();
}


void DatasetInterface::add_training_for_regression(sDatasetItem &item)
{
  if (training.size() != 1)
  {
    training.resize(1);
  }

  training[0].push_back(item);

  output_size   = training.size();
  training_size = training[0].size();
}

void DatasetInterface::add_testing(sDatasetItem &item)
{
  testing.push_back(item);
}

void DatasetInterface::add_unlabeled(sDatasetItem &item)
{
  unlabeled.push_back(item);
}

unsigned int DatasetInterface::argmax(std::vector<float> &v)
{
  unsigned int result = 0;

  for (unsigned int i = 0; i < v.size(); i++)
    if (v[i] > v[result])
      result = i;

  return result;
}


void DatasetInterface::print_testing_item(unsigned int idx)
{
  printf("\n");
  for (unsigned int i = 0; i < testing[idx].output.size(); i++)
    printf("%5.6f ", testing[idx].output[i]);
  printf(" :\n");

  unsigned int ptr = 0;
  for (unsigned int ch = 0; ch < channels; ch++)
  {
    for (unsigned int y = 0; y < height; y++)
    {
      for (unsigned int x = 0; x < width; x++)
      {
        float v = testing[idx].input[ptr];
        printf("%5.6f ", v);

        ptr++;
      }
      printf("\n");
    }
    printf("\n");
  }
}

void DatasetInterface::export_h_testing(std::string file_name, unsigned int count)
{
  Log result(file_name);

  result << "#ifndef _DATASET_H_\n";
  result << "#define _DATASET_H_\n";
  result << "\n";

  result << "#define DATASET_COUNT (unsigned int)(" << count << ")\n";
  result << "\n";

  result << "#define DATASET_WIDTH  (unsigned int)(" << width << ")\n";
  result << "#define DATASET_HEIGHT  (unsigned int)(" << height << ")\n";
  result << "#define DATASET_CHANNELS  (unsigned int)(" << channels << ")\n";
  result << "\n";
  result << "\n";

  result << "const unsigned char dataset[]={";

  for (unsigned int j = 0; j < count; j++)
  {
    for (unsigned int i = 0; i < get_input_size(); i++)
    {
      if ((i%16) == 0)
        result << "\n";
      unsigned char v = testing[j].input[i]*255;
      result << v << ", ";
    }

    result << "\n";
  }
  result << "};\n\n\n";


  result << "const unsigned int labels[]={";

  for (unsigned int j = 0; j < count; j++)
  {
    if ((j%16) == 0)
      result << "\n";
    result << argmax(testing[j].output) << ", ";
  }
  result << "};\n\n\n";

  result << "#endif\n";
}


void DatasetInterface::save_to_json(std::string file_name)
{
  JsonConfig json;

  unsigned int classes_count = (int)training.size();

  json.result["width"]    = width;
  json.result["height"]   = height;
  json.result["channels"] = channels;
  json.result["classes_count"] = classes_count;

  std::vector<unsigned char> tmp_input(width*height*channels);
  std::vector<unsigned char> tmp_output(classes_count);

  for (unsigned int j = 0; j < testing.size(); j++)
  {
    for (unsigned int i = 0; i < testing[j].input.size(); i++)
      json.result["testing"][j]["input"][i]  = (int)(testing[j].input[i]*255);

    for (unsigned int i = 0; i < testing[j].output.size(); i++)
      json.result["testing"][j]["output"][i]  = (int)(testing[j].output[i]*255);
  }

  unsigned int idx = 0;
  for (unsigned int class_id = 0; class_id < training.size(); class_id++)
  for (unsigned int j = 0; j < training[class_id].size(); j++)
  {
    for (unsigned int i = 0; i < training[class_id][j].input.size(); i++)
      json.result["training"][idx]["input"][i]  = (int)(training[class_id][j].input[i]*255);

    for (unsigned int i = 0; i < training[class_id][j].output.size(); i++)
      json.result["training"][idx]["output"][i]  = (int)(training[class_id][j].output[i]*255);

    idx++;
  }


  json.save(file_name);
}


void DatasetInterface::save_to_txt_training(std::string file_name)
{
  std::string result;

  std::ofstream file;
  file.open(file_name, std::ios::app);



  for (unsigned int class_id = 0; class_id < training.size(); class_id++)
  for (unsigned int j = 0; j < training[class_id].size(); j++)
  {
    std::stringstream result;

    for (unsigned int i = 0; i < training[class_id][j].input.size(); i++)
      result << std::fixed << std::setprecision(3) << training[class_id][j].input[i] << " ";

    result << argmax(training[class_id][j].output) << "\n";
    file << result.rdbuf();
  }
}


void DatasetInterface::save_to_txt_testing(std::string file_name)
{
  std::string result;

  std::ofstream file;
  file.open(file_name, std::ios::app);

  for (unsigned int j = 0; j < testing.size(); j++)
  {
    std::stringstream result;

    for (unsigned int i = 0; i < testing[j].input.size(); i++)
      result << std::fixed << std::setprecision(3) << testing[j].input[i] << " ";

    result << argmax(testing[j].output) << "\n";
    file << result.rdbuf();
  }
}


void DatasetInterface::save_to_binary(  std::string training_file_name,
                                        std::string testing_file_name,
                                        std::string unlabeled_file_name)
{
  {
    std::ofstream file;
    file.open (training_file_name, std::ios::out | std::ios::binary);

    make_header(file, get_training_size());

    for (unsigned int j = 0; j < training.size(); j++)
      for (unsigned int i = 0; i < training[j].size(); i++)
      {
        save_item(file, training[j][i]);
      }
  }

  {
    std::ofstream file;
    file.open (testing_file_name, std::ios::out | std::ios::binary);

    make_header(file, get_training_size());

    for (unsigned int j = 0; j < testing.size(); j++)
    {
      save_item(file, testing[j]);
    } 
  }

  {
    std::ofstream file;
    file.open (unlabeled_file_name, std::ios::out | std::ios::binary);

    make_header(file, get_training_size());

    for (unsigned int j = 0; j < unlabeled.size(); j++)
    {
      save_item(file, unlabeled[j]);
    }
  }
}

void DatasetInterface::save_item(std::ofstream &file, sDatasetItem &item)
{
  file.write(reinterpret_cast<const char *>(&item.input[0]), item.input.size()*sizeof(float));
  file.write(reinterpret_cast<const char *>(&item.output[0]), item.output.size()*sizeof(float));
}

void DatasetInterface::make_header(std::ofstream &file, unsigned int items_count)
{
  unsigned int magic = get_binary_magic();

  file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  file.write(reinterpret_cast<const char *>(&items_count), sizeof(items_count));
  file.write(reinterpret_cast<const char *>(&width), sizeof(width));
  file.write(reinterpret_cast<const char *>(&height), sizeof(height));
  file.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
  file.write(reinterpret_cast<const char *>(&output_size), sizeof(output_size));
}
