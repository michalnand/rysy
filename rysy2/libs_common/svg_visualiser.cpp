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

void SVGVisualiser::process(std::string image_file_name, Shape input_shape)
{
  int layer_width  = 50;
  int layer_height = 500;

  int layer_offset_y = 0;
  int layer_offset_x = 0;

  float layer_spacing = 1.8;

  if (json_config["layers"].size() > 16)
    layer_spacing = 1.3;

  unsigned int width = layer_width*layer_spacing*(1+json_config["layers"].size());
  unsigned int height = layer_height;

  svg.add_header(width, height);


  for (int layer = -1; layer < (int)json_config["layers"].size(); layer++)
  {
    int y = 1 + layer_offset_y;
    int x = (layer + 1)*(layer_width*layer_spacing) + layer_offset_x;

    int r = 100;
    int g = 100;
    int b = 100;


    if (layer == -1)
    {
      int r = 200;
      int g = 200;
      int b = 0;

      int sr = r/2;
      int sg = g/2;
      int sb = b/2;


      svg.add_rectangle(x, y, layer_width, layer_height,
                    2,
                    r, g, b,
                    sr, sg, sb);

      unsigned int font_size = 30;

      std::string layer_name = "input ";

      layer_name+= std::to_string(input_shape.w()) + " ";
      layer_name+= std::to_string(input_shape.h()) + " ";
      layer_name+= std::to_string(input_shape.d()) + " ";

                    svg.add_text(x + layer_width/2, y + layer_height/2 - font_size*layer_name.size()/4, layer_name,
                            font_size,
                            true);
    }
    else
    {
        std::string layer_type = json_config["layers"][layer]["type"].asString();

        if (layer_type == "convolution")
        {
          r = 80;
          g = 80;
          b = 190;
        }

        if ((layer_type == "dense convolution")||(layer_type == "dense_convolution"))
        {
          r = 120;
          g = 80;
          b = 190;
        }

        if ((layer_type == "fc")||(layer_type == "output"))
        {
          r = 190;
          g = 190;
          b = 80;
        }

        if ( (layer_type == "max_pooling") ||
             (layer_type == "max pooling") ||
             (layer_type == "average pooling") ||
             (layer_type == "average_pooling") ||
             (layer_type == "unpooling"))
        {
          r = 80;
          g = 190;
          b = 80;
        }


        if ((layer_type == "relu") ||
           (layer_type == "elu") )
        {
          r = 190;
          g = 80;
          b = 80;
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
          std::string shape = " ";
          shape+= json_config["output_shape"][0].asString() + " ";
          shape+= json_config["output_shape"][1].asString() + " ";
          shape+= json_config["output_shape"][2].asString() + " ";
          layer_name+= shape;
        }
        else
        {
            if (json_config["layers"][layer]["shape"][0].asInt() != 0)
            {
              std::string shape = " ";

              if (json_config["layers"][layer]["shape"][0].asInt() > 0)
                shape+= json_config["layers"][layer]["shape"][0].asString() + " ";

              if (json_config["layers"][layer]["shape"][1].asInt() > 0)
                shape+= json_config["layers"][layer]["shape"][1].asString() + " ";

              if (json_config["layers"][layer]["shape"][2].asInt() > 0)
                shape+= json_config["layers"][layer]["shape"][2].asString() + " ";
              layer_name+= shape;
            }
        }

        svg.add_text(x + layer_width/2, y + layer_height/2 - font_size*layer_name.size()/4, layer_name,
                font_size,
                true);
    }
  }

  svg.add_footer();

  svg.save(image_file_name);
}
