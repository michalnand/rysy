#ifndef _DATASET_IMAGES_H_
#define _DATASET_IMAGES_H_

#include "dataset_interface.h"

#include <json_config.h>

class DatasetImages: public DatasetInterface
{
  private:
    bool grayscale;
    float testing_ratio;
    int max_items_per_folder;

  public:
    DatasetImages(std::string json_config_file_name);
    ~DatasetImages();

  private:
    void load(Json::Value parameters, unsigned int classes_count);
    void load_dir(std::string path, unsigned int class_id, unsigned int classes_count);


};

#endif
