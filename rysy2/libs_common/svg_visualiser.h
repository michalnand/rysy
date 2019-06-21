#ifndef _SVG_VISUALISER_H_
#define _SVG_VISUALISER_H_

#include <string.h>
#include <json_config.h>
#include <svg.h>

class SVGVisualiser
{
  public:
    Json::Value json_config;
    SVG svg;

  public:
    SVGVisualiser(std::string json_file_name);
    SVGVisualiser(Json::Value &json_config);

    void process(std::string image_file_name);

  private:
    void init(Json::Value &json_config_);
};

#endif
