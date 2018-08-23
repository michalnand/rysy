#include "image_save.h"

#include <CImg.h>


ImageSave::ImageSave(unsigned int width, unsigned int height, bool grayscale, bool display_enabled)
{
  m_width           = width;
  m_height          = height;
  m_grayscale       = grayscale;
  m_display_enabled = display_enabled;

  output_image = nullptr;
  image_display = nullptr;

  output_image = new cimg_library::CImg<float>(m_width, m_height, 1, 3, 0);

  if (m_display_enabled)
  {
    image_display = new cimg_library::CImgDisplay(*output_image, "image", 0);
  }
}

ImageSave::~ImageSave()
{
  if (image_display != nullptr)
    delete image_display;

  if (output_image != nullptr)
    delete output_image;
}

void ImageSave::save(std::string file_name, std::vector<float> &v)
{
  vector_to_image(v);
  output_image->save(file_name.c_str());
}

void ImageSave::show(std::vector<float> &v)
{
  vector_to_image(v);


  if (image_display != nullptr)
    delete image_display;

  image_display = new cimg_library::CImgDisplay(*output_image, "image", 0);
}


void ImageSave::vector_to_image(std::vector<float> &v)
{
  std::vector<float> pixel(3);

  if (m_grayscale)
  {
    for (unsigned int y = 0; y < m_height; y++)
    for (unsigned int x = 0; x < m_width; x++)
    {
      pixel[0] = v[(0*m_height + y)*m_width + x]*255;
      pixel[1] = v[(0*m_height + y)*m_width + x]*255;
      pixel[2] = v[(0*m_height + y)*m_width + x]*255;

      output_image->draw_point(x, y, &pixel[0]);
    }
  }
  else
  {
    for (unsigned int y = 0; y < m_height; y++)
    for (unsigned int x = 0; x < m_width; x++)
    {
      pixel[0] = v[(0*m_height + y)*m_width + x]*255;
      pixel[1] = v[(1*m_height + y)*m_width + x]*255;
      pixel[2] = v[(2*m_height + y)*m_width + x]*255;

      output_image->draw_point(x, y, &pixel[0]);
    }
  }
}
