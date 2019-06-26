#ifndef _LAYER_EXPORT_H_
#define _LAYER_EXPORT_H_

#include <string>
#include <json_config.h>

#include <shape.h>

class LayerExport
{
  private:
    Json::Value json;
    sShape input_shape;
    sShape output_shape;

    std::string layer_prefix;
    std::string result;
    std::string export_path;

  public:
    LayerExport(  std::string export_path,
                  Json::Value &json,
                  std::string layer_prefix,
                  sShape input_shape,
                  sShape output_shape);
    virtual ~LayerExport();

    std::string& get()
    {
      return result;
    }

    void save();

  private:
    void process();
    void raw_float_to_array(std::string &result, std::string prefix, std::string file_name);
    std::vector<float> read_raw(std::string file_name);
    float find_min(std::vector<float> &v);
    float find_max(std::vector<float> &v);

};

#endif
