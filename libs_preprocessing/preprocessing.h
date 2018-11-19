#ifndef _PREPROCESSING_H_
#define _PREPROCESSING_H_

#include <vector>
#include <json_config.h>
#include <preprocessing_layer.h>

class Preprocessing
{
  protected:
    std::vector<Tensor*> layers_output;
    std::vector<PreprocessingLayer*> layers;

  public:
    Preprocessing();
    Preprocessing(Json::Value json, sGeometry input_geometry = {0, 0, 0});
    Preprocessing(std::string json_file_name, sGeometry input_geometry = {0, 0, 0});
    virtual ~Preprocessing();

    void init(Json::Value json, sGeometry input_geometry);

    void process(Tensor &output, Tensor &input, unsigned int augumentation = 0);
};

#endif
