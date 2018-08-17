#ifndef _CS_DATASET_CREATE_H_
#define _CS_DATASET_CREATE_H_


#include "cs_parse_file.h"


class CSDatasetCreate
{
  private:
    Json::Value json_config;


    unsigned int output_image_size;
    unsigned int image_step;
    unsigned int images_count;
    bool random_mode;
    std::string output_dir;

  public:
    CSDatasetCreate(std::string json_file_name);
    CSDatasetCreate(Json::Value json_config);

    virtual ~CSDatasetCreate();

    void process();
    
  private:
    void init(Json::Value json_config);

    bool file_exists(std::string file_name);

};


#endif
