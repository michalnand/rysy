#ifndef _PREPROCESSING_LAYER_H_
#define _PREPROCESSING_LAYER_H_

#include <json_config.h>
#include <tensor.h>
#include <network_config.h>


class PreprocessingLayer
{
  protected:
    Json::Value parameters;

  public:
    PreprocessingLayer();
    PreprocessingLayer(Json::Value parameters);
    PreprocessingLayer(PreprocessingLayer& other);
    PreprocessingLayer(const PreprocessingLayer& other);

    virtual ~PreprocessingLayer();

    PreprocessingLayer& operator= (PreprocessingLayer& other);
    PreprocessingLayer& operator= (const PreprocessingLayer& other);

  protected:
    void copy(PreprocessingLayer& other);
    void copy(const PreprocessingLayer& other);

  public:
    virtual void process(Tensor &output, Tensor &input);

    virtual sGeometry get_output_geometry(sGeometry input_geometry);
};


#endif
