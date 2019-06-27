#include <layer_export.h>

#include <fstream>
#include <iostream>
#include <math.h>

#include <log.h>

LayerExport::LayerExport( std::string export_path,
                          Json::Value &json,
                          std::string layer_prefix,
                          sEmbeddedNetShape input_shape,
                          sEmbeddedNetShape output_shape)
{
  this->layer_prefix = layer_prefix;
  this->json = json;
  this->input_shape = input_shape;
  this->output_shape = output_shape;
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

  sEmbeddedNetShape shape;


  shape.w = json["shape"][0].asInt();
  shape.h = json["shape"][1].asInt();
  shape.d = json["shape"][2].asInt();

  if ((shape.w == 0)&&(shape.h == 0)&&(shape.d == 0))
  {
    shape = input_shape;
  }

  result+= "#ifndef _LAYER_" + layer_prefix + "_H_\n";
  result+= "#define _LAYER_" + layer_prefix + "_H_\n";
  result+= "\n\n";
  result+= "#include <EmbeddedNetConfig.h>\n";
  result+= "\n\n";

  result+= "#define " + layer_prefix + "_type " + "\"" + json["type"].asString() + "\"\n";

  result+= "\n";

  result+= "const sEmbeddedNetShape " + layer_prefix + "_input_shape = {";
  result+= std::to_string(input_shape.w) + ", ";
  result+= std::to_string(input_shape.h) + ", ";
  result+= std::to_string(input_shape.d) + "};\n";

  result+= "const sEmbeddedNetShape " + layer_prefix + "_output_shape = {";
  result+= std::to_string(output_shape.w) + ", ";
  result+= std::to_string(output_shape.h) + ", ";
  result+= std::to_string(output_shape.d) + "};\n";

  result+= "const sEmbeddedNetShape " + layer_prefix + "_shape = {";
  result+= std::to_string(shape.w) + ", ";
  result+= std::to_string(shape.h) + ", ";
  result+= std::to_string(shape.d) + "};\n";

  result+= "\n";
  raw_float_to_array(result, layer_prefix+"_weights", weights_path+"_weights.bin");

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
