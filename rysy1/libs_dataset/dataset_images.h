#ifndef _DATASET_IMAGES_H_
#define _DATASET_IMAGES_H_

#include "dataset_interface.h"

#include <json_config.h>
#include <mutex>

class DatasetImages: public DatasetInterface
{
  private:
    bool grayscale;
    int max_items_per_folder;

    std::mutex mutex;

  public:
    DatasetImages(std::string json_config_file_name);
    ~DatasetImages();

  private:
    void load(Json::Value parameters, unsigned int classes_count, unsigned int load_threads_count = 4, float testing_ratio = 0.1);
    void load_dir(std::string path, unsigned int class_id, unsigned int classes_count, float testing_ratio);


};

#endif
