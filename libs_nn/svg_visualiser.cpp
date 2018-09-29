#include "svg_visualiser.h"
#include <stdio.h>


SVGVisualiser::SVGVisualiser(std::string json_file_name)
{
  JsonConfig json(json_file_name);
  init(json.result);
}

SVGVisualiser::SVGVisualiser(Json::Value &json_config)
{
  init(json_config);
}

void SVGVisualiser::init(Json::Value &json_config_)
{
  this->json_config = json_config_;
}

void SVGVisualiser::process(std::string image_file_name)
{
  int layer_width  = 50;
  int layer_height = 500;

  int layer_offset_y = 0;
  int layer_offset_x = 0;

  unsigned int width = layer_width*2*(1+json_config["layers"].size());
  unsigned int height = layer_height;

  svg.add_header(width, height);


  for (int layer = -1; layer < (int)json_config["layers"].size(); layer++)
  {
    int y = 1 + layer_offset_y;
    int x = (layer + 1)*(layer_width*2) + layer_offset_x;

    int r = 100;
    int g = 100;
    int b = 100;


    if (layer == -1)
    {
      int r = 200;
      int g = 80;
      int b = 80;

      int sr = r/2;
      int sg = g/2;
      int sb = b/2;


      svg.add_rectangle(x, y, layer_width, layer_height,
                    2,
                    r, g, b,
                    sr, sg, sb);

      unsigned int font_size = 30;

      std::string layer_name = "input ";

      layer_name+= json_config["input_geometry"][0].asString() + " ";
      layer_name+= json_config["input_geometry"][1].asString() + " ";
      layer_name+= json_config["input_geometry"][2].asString() + " ";

                    svg.add_text(x + layer_width/2, y + layer_height/2 - font_size*layer_name.size()/4, layer_name,
                            font_size,
                            true);
    }
    else
    {
        std::string layer_type = json_config["layers"][layer]["type"].asString();

        if (layer_type == "convolution")
        {
          r = 100;
          g = 100;
          b = 190;
        }

        if ((layer_type == "dense convolution")||(layer_type == "dense_convolution"))
        {
          r = 100;
          g = 100;
          b = 190;
        }

        if (layer_type == "gating")
        {
          r = 0;
          g = 190;
          b = 0;
        }

        if (layer_type == "max_pooling")
        {
          r = 190;
          g = 100;
          b = 100;
        }

        if (layer_type == "unpooling")
        {
          r = 100;
          g = 190;
          b = 100;
        }

        if (layer_type == "fc")
        {
          r = 100;
          g = 190;
          b = 190;
        }

        if ((layer_type == "relu") ||
           (layer_type == "lrelu") ||
           (layer_type == "tanh") )
        {
          r = 190;
          g = 190;
          b = 100;
        }

        int sr = r/2;
        int sg = g/2;
        int sb = b/2;

        int font_size = 30;

        svg.add_rectangle(x, y, layer_width, layer_height,
                      2,
                      r, g, b,
                      sr, sg, sb);

        std::string layer_name = json_config["layers"][layer]["type"].asString();

        if (layer_name == "output")
        {
          std::string geometry = " ";
          geometry+= json_config["output_geometry"][0].asString() + " ";
          geometry+= json_config["output_geometry"][1].asString() + " ";
          geometry+= json_config["output_geometry"][2].asString() + " ";
          layer_name+= geometry;
        }
        else
        {
          std::string geometry = " ";
          geometry+= json_config["layers"][layer]["geometry"][0].asString() + " ";
          geometry+= json_config["layers"][layer]["geometry"][1].asString() + " ";
          geometry+= json_config["layers"][layer]["geometry"][2].asString() + " ";
          layer_name+= geometry;
        }

        svg.add_text(x + layer_width/2, y + layer_height/2 - font_size*layer_name.size()/4, layer_name,
                font_size,
                true);
    }
  }

  svg.add_footer();

  svg.save(image_file_name);
}
