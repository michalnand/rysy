#include "cs_parse_file.h"
#include <iostream>

#include <CImg.h>


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
  std::string input_image_file_name = input_file_name_prefix + "leftImg8bit.png";
  cimg_library::CImg<float> input_image(input_image_file_name.c_str());

  cimg_library::CImg<float> output_image(output_image_size, output_image_size, 1, 3, 0);

  std::vector<float> pixel(3);

  std::string output_file_name;

  unsigned int id = 0;

  input_image.normalize(0, 255);

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
        int y_ = y + ky - output_image_size/2;
        int x_ = x + kx - output_image_size/2;

        pixel[0] = *(input_image.data(x_, y_, 0, 0));
        pixel[1] = *(input_image.data(x_, y_, 0, 1));
        pixel[2] = *(input_image.data(x_, y_, 0, 2));

        output_image.draw_point(kx, ky, &pixel[0]);
      }


      output_file_name = ouput_images_dir + std::to_string(class_id) + "/" + ouput_images_prefix + std::to_string(id) + ".png";
      // std::cout << "saving to " << ouput_images_dir << "\n\n\n";
      output_image.save(output_file_name.c_str());

      id++;
    }
  }

}



void CSParseFile::process_random(unsigned int count)
{
  std::string input_image_file_name = input_file_name_prefix + "leftImg8bit.png";
  cimg_library::CImg<float> input_image(input_image_file_name.c_str());

  cimg_library::CImg<float> output_image(output_image_size, output_image_size, 1, 3, 0);

  std::vector<float> pixel(3);

  std::string output_file_name;

  unsigned int id = 0;

  input_image.normalize(0, 255);

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
        int y_ = y + ky - output_image_size/2;
        int x_ = x + kx - output_image_size/2;

        pixel[0] = *(input_image.data(x_, y_, 0, 0));
        pixel[1] = *(input_image.data(x_, y_, 0, 1));
        pixel[2] = *(input_image.data(x_, y_, 0, 2));

        output_image.draw_point(kx, ky, &pixel[0]);
      }


      output_file_name = ouput_images_dir + std::to_string(class_id) + "/" + ouput_images_prefix + std::to_string(id) + ".png";
      // std::cout << "saving to " << ouput_images_dir << "\n\n\n";

      output_image.save(output_file_name.c_str());
      id++;
    }
  }
}
