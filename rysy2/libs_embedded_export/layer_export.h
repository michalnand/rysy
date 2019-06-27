#ifndef _LAYER_EXPORT_H_
#define _LAYER_EXPORT_H_

#include <string>
#include <json_config.h>

#include <EmbeddedNetConfig.h>

class LayerExport
{
  private:
    Json::Value json;
    sEmbeddedNetShape input_shape;
    sEmbeddedNetShape output_shape;

    std::string layer_prefix;
    std::string result;
    std::string export_path;

  public:
    LayerExport(  std::string export_path,
                  Json::Value &json,
                  std::string layer_prefix,
                  sEmbeddedNetShape input_shape,
                  sEmbeddedNetShape output_shape);
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
