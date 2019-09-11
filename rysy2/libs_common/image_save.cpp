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

void ImageSave::save(std::string file_name, std::vector<std::vector<float>> &v)
{
    m_grayscale = true;

    unsigned int height = v.size();
    unsigned int width  = v[0].size();
    unsigned int size = width*height;

    std::vector<float> v_tmp(size);

    unsigned int idx = 0;
    for (unsigned int y = 0; y < height; y++)
        for (unsigned int x = 0; x < width; x++)
        {
            v_tmp[idx] = v[y][x];
            idx++;
        }

    save(file_name, v_tmp);
}

void ImageSave::save(std::string file_name, std::vector<std::vector<std::vector<float>>> &v)
{
    m_grayscale = false;

    unsigned int channels   = 3;
    unsigned int height     = v[0].size();
    unsigned int width      = v[0][0].size();

    unsigned int size = width*height*channels;

    std::vector<float> v_tmp(size);

    unsigned int idx = 0;
    for (unsigned int ch = 0; ch < channels; ch++)
        for (unsigned int y = 0; y < height; y++)
            for (unsigned int x = 0; x < width; x++)
            {
                v_tmp[idx] = v[ch][y][x];
                idx++;
            }

    save(file_name, v_tmp);
}

void ImageSave::show(std::vector<float> &v, std::string window_name)
{
  vector_to_image(v);


  if (image_display != nullptr)
    delete image_display;

  image_display = new cimg_library::CImgDisplay(*output_image, window_name.c_str(), 0);
}


void ImageSave::vector_to_image(std::vector<float> v)
{
  std::vector<float> pixel(3);

  normalise(v, 0, 255);

  if (m_grayscale)
  {
    for (unsigned int y = 0; y < m_height; y++)
    for (unsigned int x = 0; x < m_width; x++)
    {
      pixel[0] = v[(0*m_height + y)*m_width + x];
      pixel[1] = v[(0*m_height + y)*m_width + x];
      pixel[2] = v[(0*m_height + y)*m_width + x];

      output_image->draw_point(x, y, &pixel[0]);
    }
  }
  else
  {
    for (unsigned int y = 0; y < m_height; y++)
    for (unsigned int x = 0; x < m_width; x++)
    {
      pixel[0] = v[(0*m_height + y)*m_width + x];
      pixel[1] = v[(1*m_height + y)*m_width + x];
      pixel[2] = v[(2*m_height + y)*m_width + x];

      output_image->draw_point(x, y, &pixel[0]);
    }
  }
}


void ImageSave::normalise(std::vector<float> &v, float min, float max)
{
  float max_v = v[0];
  float min_v = v[0];
  for (unsigned int i = 0; i < v.size(); i++)
  {
    if (v[i] > max_v)
      max_v = v[i];

    if (v[i] < min_v)
      min_v = v[i];
  }

  float k = 0.0;
  float q = 0.0;

  if (max_v > min_v)
  {
    k = (max - min)/(max_v - min_v);
    q = max - k*max_v;
  }

  for (unsigned int i = 0; i < v.size(); i++)
  {
    v[i] = k*v[i] + q;
  }
}
