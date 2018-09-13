#include "layer_export.h"

#include <fstream>
#include <iostream>
#include <math.h>

#include <log.h>

LayerExport::LayerExport( std::string export_path,
                          Json::Value &json,
                          std::string layer_prefix,
                          sGeometry input_geometry,
                          sGeometry output_geometry)
{
  this->layer_prefix = layer_prefix;
  this->json = json;
  this->input_geometry = input_geometry;
  this->output_geometry = output_geometry;
  this->export_path = export_path;

  process();
  save();
}

LayerExport::~LayerExport()
{

}

void LayerExport::save()
{
  std::cout << "saving layer to " << export_path+layer_prefix+".h" << "\n";
  Log output_file(export_path+layer_prefix+".h");
  output_file << result;
}

void LayerExport::process()
{
  std::string weights_path = json["weights_file_name"].asString();

  sGeometry kernel_geometry;


  kernel_geometry.w = json["geometry"][0].asInt();
  kernel_geometry.h = json["geometry"][1].asInt();
  kernel_geometry.d = json["geometry"][2].asInt();

  if ((kernel_geometry.w == 0)||(kernel_geometry.h == 0)||(kernel_geometry.d == 0))
  {
    kernel_geometry = input_geometry;
  }

  result+= "#ifndef _LAYER_" + layer_prefix + "_H_\n";
  result+= "#define _LAYER_" + layer_prefix + "_H_\n";
  result+= "\n\n";
  result+= "#include <NetworkConfig.h>\n";
  result+= "\n\n";

  result+= "#define " + layer_prefix + "_type " + "\"" + json["type"].asString() + "\"\n";

  result+= "\n";

  result+= "const sLayerGeometry " + layer_prefix + "_input_geometry = {";
  result+= std::to_string(input_geometry.w) + ", ";
  result+= std::to_string(input_geometry.h) + ", ";
  result+= std::to_string(input_geometry.d) + "};\n";

  result+= "const sLayerGeometry " + layer_prefix + "_output_geometry = {";
  result+= std::to_string(output_geometry.w) + ", ";
  result+= std::to_string(output_geometry.h) + ", ";
  result+= std::to_string(output_geometry.d) + "};\n";

  result+= "const sLayerGeometry " + layer_prefix + "_kernel_geometry = {";
  result+= std::to_string(kernel_geometry.w) + ", ";
  result+= std::to_string(kernel_geometry.h) + ", ";
  result+= std::to_string(kernel_geometry.d) + "};\n";

  result+= "\n";
  raw_float_to_array(result, layer_prefix+"_weights", weights_path+"_weight.bin");

  result+= "\n\n\n";
  raw_float_to_array(result, layer_prefix+"_bias", weights_path+"_bias.bin");


  result+="\n#endif\n";
}





void LayerExport::raw_float_to_array(std::string &result, std::string prefix, std::string file_name)
{
  auto raw = read_raw(file_name);

  if (raw.size() > 0)
  {
    auto min = find_min(raw);
    auto max = find_max(raw);

    float range;

    if (fabs(min) > fabs(max))
      range = fabs(min);
    else
      range = fabs(max);

    int range_int = range*1024;

    result+= "#define " + prefix + "_size ((unsigned int)" + std::to_string(raw.size()) + ") //array size\n";
    result+= "#define " + prefix + "_range ((nn_t)" + std::to_string(range_int) + ") //multiply neuron result with range/1024\n";

    result+= "\n";
    result+= "const nn_weight_t " + prefix + "[]={\n";

    for (unsigned int i = 0; i < raw.size(); i++)
    {
      int value = (raw[i]/range)*127.0;
      result+= std::to_string(value) + ", ";

      if (((i+1)%16) == 0)
        result+= "\n";
    }

    result+= "};\n";

    result+="\n";
  }
}

std::vector<float> LayerExport::read_raw(std::string file_name)
{
  std::vector<float> result;
  float f = 0.0;

  std::ifstream fin(file_name, std::ios::binary);

  std::cout << "loading weights from " << file_name << "\n";


  if (fin.is_open())
  {
    while (fin.read(reinterpret_cast<char*>(&f), sizeof(float)))
      result.push_back(f);
  }


  return result;
}

float LayerExport::find_min(std::vector<float> &v)
{
  float result = v[0];

  for (unsigned int i = 0; i < v.size(); i++)
    if (v[i] < result)
      result = v[i];

  return result;
}

float LayerExport::find_max(std::vector<float> &v)
{
  float result = v[0];

  for (unsigned int i = 0; i < v.size(); i++)
    if (v[i] > result)
      result = v[i];

  return result;
}
