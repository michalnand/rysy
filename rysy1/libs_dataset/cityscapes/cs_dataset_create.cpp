#include "cs_dataset_create.h"
#include <iostream>
#include <fstream>

#include <experimental/filesystem>


CSDatasetCreate::CSDatasetCreate(std::string json_file_name)
{
  JsonConfig json(json_file_name);
  init(json.result);
}

CSDatasetCreate::CSDatasetCreate(Json::Value json_config)
{
  init(json_config);
}

CSDatasetCreate::~CSDatasetCreate()
{

}


void CSDatasetCreate::init(Json::Value json_config)
{
  this->json_config = json_config;


  output_image_size = this->json_config["output_image_size"].asInt();
  image_step        = this->json_config["image_step"].asInt();
  images_count      = this->json_config["images_count"].asInt();
  random_mode       = this->json_config["random_mode"].asBool();
  output_dir        = this->json_config["output_dir"].asString();
}

void CSDatasetCreate::process()
{
  unsigned int dir_count = this->json_config["input_dirs_images"].size();

  for (unsigned int dir = 0; dir < dir_count; dir++)
  {
    std::string image_dir = this->json_config["input_dirs_images"][dir].asString();
    std::string label_dir = this->json_config["input_dirs_labels"][dir].asString();

    std::cout << "image dir  " << image_dir << "\n";
    std::cout << "label dir  " << label_dir << "\n";
    std::cout << "done " << dir*100.0/dir_count << "%\n\n";

    unsigned int file_id = 0;
    for (auto & p : std::experimental::filesystem::directory_iterator(image_dir))
    {
      std::string output_file_name_prefix = std::to_string(dir) + "_" + std::to_string(file_id) + "_";

       std::string image_file_name;
       image_file_name = p.path();

       unsigned int pos = image_file_name.find("leftImg8bit.png");
       std::string input_file_name_prefix = image_file_name.erase(pos, image_file_name.size());


       std::string file_name = std::experimental::filesystem::path(input_file_name_prefix).filename();

       std::string input_labels_file_name_prefix = label_dir + file_name;

       std::cout << "input_file_name_prefix        : " << input_file_name_prefix << "\n";
       std::cout << "input_labels_file_name_prefix : " << input_labels_file_name_prefix << "\n";



       CSParseFile file_parse( input_file_name_prefix,
                               input_labels_file_name_prefix,
                               output_dir,
                               output_file_name_prefix ,
                               output_image_size);

       if (random_mode)
          file_parse.process_random(images_count);
       else
          file_parse.process_all(image_step);

       file_id++;
     }
  }
}

bool CSDatasetCreate::file_exists(std::string file_name)
{
    std::ifstream f(file_name.c_str());
    return f.good();
}
