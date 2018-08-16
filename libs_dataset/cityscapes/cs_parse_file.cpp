#include "cs_parse_file.h"
#include <image.h>

CSParseFile::CSParseFile(
                          std::string input_file_name_prefix,
                          std::string ouput_images_dir,
                          std::string ouput_images_prefix,
                          unsigned int output_image_size)
{
  this->input_file_name_prefix  = input_file_name_prefix;

  this->ouput_images_dir        = ouput_images_dir;
  this->ouput_images_prefix     = ouput_images_prefix;
  this->output_image_size       = output_image_size;


  labels.load(this->input_file_name_prefix + "gtCoarse_polygons.json");

  cs_to_class_id.load("input/label_names.json");
}

CSParseFile::~CSParseFile()
{

}


void CSParseFile::process_all(unsigned int image_step)
{
  Image input_image(input_file_name_prefix + "leftImg8bit.png");
  Image output_image(output_image_size, output_image_size);

  std::string output_file_name;

  unsigned int id = 0;
  for (unsigned int y = (output_image_size/2); y < (labels.height() - (output_image_size/2)); y+= image_step)
    for (unsigned int x = (output_image_size/2); x < (labels.width() - (output_image_size/2)); x+= image_step)
    {
      auto class_name = labels.get_label(x, y, 0);
      int class_id = cs_to_class_id.get(class_name);

      if (class_id != -1)
      {
        for (unsigned int ky = 0; ky < output_image_size; ky++)
        for (unsigned int kx = 0; kx < output_image_size; kx++)
        {
          output_image.pixels[ky][kx] = input_image.pixels[y + ky - output_image_size/2][x + kx - output_image_size/2];
        }

        output_file_name = ouput_images_dir + std::to_string(class_id) + "/" + ouput_images_prefix + std::to_string(id) + ".png";
        output_image.normalise();
        output_image.save(output_file_name);

        id++;
      }
    }
}



void CSParseFile::process_random(unsigned int count)
{
  Image input_image(input_file_name_prefix + "leftImg8bit.png");
  Image output_image(output_image_size, output_image_size);

  std::string output_file_name;

  unsigned int id = 0;

  while (id < count)
  {
    unsigned int y = (output_image_size/2) + rand()%(labels.height() - output_image_size);
    unsigned int x = (output_image_size/2) + rand()%(labels.width() - output_image_size);

    auto class_name = labels.get_label(x, y, 0);
    int class_id = cs_to_class_id.get(class_name);

      if (class_id != -1)
      {
        for (unsigned int ky = 0; ky < output_image_size; ky++)
        for (unsigned int kx = 0; kx < output_image_size; kx++)
        {
          output_image.pixels[ky][kx] = input_image.pixels[y + ky - output_image_size/2][x + kx - output_image_size/2];
        }

        output_file_name = ouput_images_dir + std::to_string(class_id) + "/" + ouput_images_prefix + std::to_string(id) + ".png";
        output_image.normalise();
        output_image.save(output_file_name);

        id++;
      }
  }
}
