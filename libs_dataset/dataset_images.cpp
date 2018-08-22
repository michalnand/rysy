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

  unsigned int classes_count = json.result["classes count"].asInt();
  training.resize(classes_count);



  load(json.result["training"], classes_count, false);
  load(json.result["testing"], classes_count, true);

  print();
}

DatasetImages::~DatasetImages()
{

}



void DatasetImages::load(Json::Value parameters, unsigned int classes_count, bool testing)
{
  for (unsigned int j = 0; j < parameters.size(); j++)
    load_dir(parameters[j]["path"].asString(), parameters[j]["class"].asInt(), classes_count, testing);
}

void DatasetImages::load_dir(std::string path, unsigned int class_id, unsigned int classes_count, bool testing)
{
  printf("loading directory %s\n", path.c_str());

  int items_count = 0;

  for (auto & p : std::experimental::filesystem::directory_iterator(path))
   {
     std::string image_file_name;
     image_file_name = p.path();

     if (std::experimental::filesystem::path(image_file_name).extension() == ".png")
     {
       // printf(">>>> %s \n", image_file_name.c_str());
       ImageLoad image(image_file_name, grayscale, true);

       sDatasetItem item;

       item.input = image.get();


       item.output.resize(classes_count);
       for (unsigned int i = 0; i < item.output.size(); i++)
          item.output[i] = 0.0;

       item.output[class_id] = 1.0;


       if (testing)
        add_testing(item);
       else
        add_training(item);

       items_count++;
       if (max_items_per_folder != -1)
       {
         if (items_count > max_items_per_folder)
          return;
        }
     }
   }
}
