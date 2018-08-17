#include "dataset_images.h"
#include <experimental/filesystem>
#include <image_load.h>

DatasetImages::DatasetImages(std::string json_config_file_name)
{
  JsonConfig json(json_config_file_name);

  grayscale     = json.result["grayscale"].asBool();

  width     = json.result["width"].asInt();
  height    = json.result["height"].asInt();

  max_items_per_folder = json.result["max_items_per_folder"].asInt();

  if (grayscale)
    channels  = 1;
  else
    channels  = 3;


  load(training, json.result["training"], json.result["classes count"].asInt());
  load(testing, json.result["testing"], json.result["classes count"].asInt());

  shuffle();

  printf("training loaded %u\n", (unsigned int)training.size());
  printf("testing  loaded %u\n", (unsigned int)testing.size());

}

DatasetImages::~DatasetImages()
{

}



void DatasetImages::load(std::vector<sDatasetItem> &items, Json::Value parameters, unsigned int classes_count)
{
  for (unsigned int j = 0; j < parameters.size(); j++)
    load_dir(items, parameters[j]["path"].asString(), parameters[j]["class"].asInt(), classes_count);
}

void DatasetImages::load_dir(std::vector<sDatasetItem> &items, std::string path, unsigned int class_id, unsigned int classes_count)
{
  printf("loading directory %s\n", path.c_str());

  int items_count = 0;

  for (auto & p : std::experimental::filesystem::directory_iterator(path))
   {
     std::string image_file_name;
     image_file_name = p.path();

  //   printf("loading file %s\n", image_file_name.c_str());


     if (std::experimental::filesystem::path(image_file_name).extension() == ".png")
     {
       ImageLoad image(image_file_name, grayscale, true);

       sDatasetItem item;

       item.input = image.get();

       item.output.resize(classes_count);
       for (unsigned int i = 0; i < item.output.size(); i++)
          item.output[i] = 0.0;

       item.output[class_id] = 1.0;

       items.push_back(item);

       items_count++;
       if (max_items_per_folder != -1)
       {
         if (items_count > max_items_per_folder)
          return;
        }
     }
   }
}
