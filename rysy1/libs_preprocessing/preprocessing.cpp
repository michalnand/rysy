#include "preprocessing.h"

#include <iostream>

#include "luma_noise_layer.h"
#include "white_noise_layer.h"
#include "rgb_to_yuv_layer.h"
#include "yuv_to_rgb_layer.h"
#include "rows_sign_noise_layer.h"
#include "rows_dc_offset_remove_layer.h"
#include "augmentation_layer.h"

Preprocessing::Preprocessing()
{

}

Preprocessing::Preprocessing(Json::Value json, sGeometry input_geometry)
{
  init(json, input_geometry);
}

Preprocessing::Preprocessing(std::string json_file_name, sGeometry input_geometry)
{
  JsonConfig json(json_file_name);
  init(json.result, input_geometry);
}

void Preprocessing::init(Json::Value json, sGeometry input_geometry)
{
  for (unsigned int layer = 0; layer < layers.size(); layer++)
  {
    delete layers[layer];
  }

  for (unsigned int layer = 0; layer < layers_output.size(); layer++)
  {
    delete layers_output[layer];
  }

  layers.clear();
  layers_output.clear();

  unsigned int layers_count = json["layers"].size();

  sGeometry layer_input_geometry;

  if ( (input_geometry.w == 0) || (input_geometry.h == 0) || (input_geometry.d == 0) )
  {
    layer_input_geometry.w = json["input_geometry"][0].asInt();
    layer_input_geometry.h = json["input_geometry"][1].asInt();
    layer_input_geometry.d = json["input_geometry"][2].asInt();
  }
  else
  {
    layer_input_geometry.w = input_geometry.w;
    layer_input_geometry.h = input_geometry.h;
    layer_input_geometry.d = input_geometry.d;
  }

  for (unsigned int i = 0; i < layers_count; i++)
  {
    Json::Value json_ = json["layers"][i];

    PreprocessingLayer *layer = nullptr;


    if ( (json_["type"].asString() == "luma_noise_layer") || (json_["layer"].asString() == "luma noise layer"))
      layer = new LumaNoiseLayer(json_);
    else
    if ( (json_["type"].asString() == "white_noise_layer") || (json_["layer"].asString() == "white noise layer"))
      layer = new WhiteNoiseLayer(json_);
    else
    if ( (json_["type"].asString() == "rgb_to_yuv_layer") || (json_["layer"].asString() == "rgb to yuv layer"))
      layer = new RgbToYuvLayer(json_);
    else
    if ( (json_["type"].asString() == "yuv_to_rgb_layer") || (json_["layer"].asString() == "yuv to rgb layer"))
      layer = new YuvToRgbLayer(json_);
    else
    if ( (json_["type"].asString() == "rows_sign_noise_layer") || (json_["layer"].asString() == "rows sign noise layer"))
      layer = new RowsSignNoiseLayer(json_);
    else
    if ( (json_["type"].asString() == "rows_dc_offset_remove_layer") || (json_["layer"].asString() == "rows dc offset remove layer"))
      layer = new RowsDCOffsetRemoveLayer(json_);
    else
    if ( (json_["type"].asString() == "augmentation_layer") || (json_["layer"].asString() == "augmentation layer"))
        layer = new AugmentationLayer(json_);

    sGeometry output_geometry = layer->get_output_geometry(layer_input_geometry);

    layers.push_back(layer);
    layers_output.push_back(new Tensor(output_geometry));


    layer_input_geometry = output_geometry;
  }
}

Preprocessing::~Preprocessing()
{
  for (unsigned int layer = 0; layer < layers.size(); layer++)
  {
    delete layers[layer];
  }

  for (unsigned int layer = 0; layer < layers_output.size(); layer++)
  {
    delete layers_output[layer];
  }

}

void Preprocessing::process(Tensor &output, Tensor &input, unsigned int augumentation)
{
  unsigned int size = layers.size();

  if (size == 1)
    layers[size-1]->process(output, input, augumentation);
  else
  for (unsigned int i = 0; i < size; i++)
  {
    if (i == 0)
      layers[i]->process(*layers_output[i], input, augumentation);
    else
    if (i == (size-1))
      layers[i]->process(output, *layers_output[(size-2)], augumentation);
    else
      layers[i]->process(*layers_output[i], *layers_output[i-1], augumentation);
  }

}
