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
    Preprocessing(Json::Value json, sGeometry input_geometry = {0, 0, 0});

    virtual ~Preprocessing();

    void process(Tensor &output, Tensor &input);
};

#endif
