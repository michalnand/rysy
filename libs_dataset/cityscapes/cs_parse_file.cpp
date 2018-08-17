#include "cs_parse_file.h"
#include <image.h>
#include <iostream>

CSParseFile::CSParseFile(
                          std::string input_file_name_prefix,
                          std::string input_labels_file_name_prefix,
                          std::string ouput_images_dir,
                          std::string ouput_images_prefix,
                          unsigned int output_image_size)
{
  this->input_file_name_prefix        = input_file_name_prefix;
  this->input_labels_file_name_prefix = input_labels_file_name_prefix;

  this->ouput_images_dir        = ouput_images_dir;
  this->ouput_images_prefix     = ouput_images_prefix;
  this->output_image_size       = output_image_size;


  labels.load(this->input_labels_file_name_prefix + "gtCoarse_polygons.json");

  cs_to_class_id.load("./label_names.json");
}

CSParseFile::~CSParseFile()
{

}


void CSParseFile::process_all(unsigned int image_step)
{
  Image input_image(input_file_name_prefix + "leftImg8bit.png");
  Image output_image(output_image_size, output_image_size);

  std::string output_file_name;


  float max_value = input_image.pixels[0][0].b[0];
  float min_value = max_value;

  for (unsigned int y = 0; y < (unsigned int)labels.height(); y++)
  for (unsigned int x = 0; x < (unsigned int)labels.width(); x++)
  for (unsigned int ch = 0; ch < 3; ch++)
  {
    if (input_image.pixels[y][x].b[ch] > max_value)
      max_value = input_image.pixels[y][x].b[ch];
    if (input_image.pixels[y][x].b[ch] < min_value)
      min_value = input_image.pixels[y][x].b[ch];
  }

  float k = 0.0;
  float q = 0.0;

  if (max_value > min_value)
  {
    k = (1.0 - 0.0)/(max_value - min_value);
    q = 1.0 - k*max_value;
  }


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
        for (unsigned int ch = 0; ch < 3; ch++)
        {
          output_image.pixels[ky][kx].b[ch] = k*input_image.pixels[y + ky - output_image_size/2][x + kx - output_image_size/2].b[ch] + q;
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


  std::vector<float> max_value = input_image.pixels[0][0];
  std::vector<float> min_value = max_value;

  for (unsigned int y = 0; y < (unsigned int)labels.height(); y++)
  for (unsigned int x = 0; x < (unsigned int)labels.width(); x++)
  for (unsigned int ch = 0; ch < 3; ch++)
  {
    if (input_image.pixels[y][x].b[ch] > max_value)
      max_value[ch] = input_image.pixels[y][x].b[ch];
    if (input_image.pixels[y][x].b[ch] < min_value)
      min_value[ch] = input_image.pixels[y][x].b[ch];
  }

  std::vector<float> k(max_value.size());
  std::vector<float> q(max_value.size());

  for (unsigned int i = 0; i < max_value.size(); i++)
  {
    if (max_value[i] > min_value[i])
    {
      k[i] = (1.0 - 0.0)/(max_value[i] - min_value[i]);
      q[i] = 1.0 - k[i]*max_value[i];
    }
    else
    {
      k[i] = 0.0;
      q[i] = 0.0;
    }
  }

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
      for (unsigned int ch = 0; ch < 3; ch++)
      {
        output_image.pixels[ky][kx].b[ch] = k[ch]*input_image.pixels[y + ky - output_image_size/2][x + kx - output_image_size/2].b[ch] + q[ch];
      }

      output_file_name = ouput_images_dir + std::to_string(class_id) + "/" + ouput_images_prefix + std::to_string(id) + ".png";
      // std::cout << "saving to " << ouput_images_dir << "\n\n\n";
      output_image.normalise();
      output_image.save(output_file_name);

      id++;
    }
  }
}
