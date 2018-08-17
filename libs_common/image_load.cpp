#include "image_load.h"

#include <CImg.h>

ImageLoad::ImageLoad(std::string file_name, bool load_grayscale, bool normalise)
{
  cimg_library::CImg<float> image(file_name.c_str());


  m_width     = image.width();
  m_height    = image.height();
  m_grayscale = load_grayscale;


  if (load_grayscale)
  {
    m_pixels.resize(width()*height());
  }
  else
  {
    m_pixels.resize(width()*height()*3);
  }

  if (load_grayscale)
  {
    for (unsigned int y = 0; y < height(); y++)
      for (unsigned int x = 0; x < width(); x++)
      {
        float r = *(image.data(x, y, 0, 0));
        float g = *(image.data(x, y, 0, 1));
        float b = *(image.data(x, y, 0, 2));

        float v = 0.2126*r + 0.7152*g + 0.0722*b;

        m_pixels[y*width() + x] = v;
      }
  }
  else
  {
    for (unsigned int ch = 0; ch < 3; ch++)
      for (unsigned int y = 0; y < height(); y++)
        for (unsigned int x = 0; x < width(); x++)
        {
          float v = *(image.data(x, y, 0, ch));
          m_pixels[(ch*height() + y)*width() + x] = v;
        }
  }

  if (normalise)
  {
    normalise_image(0.0, 1.0);
  }

}

ImageLoad::~ImageLoad()
{

}


std::vector<float>& ImageLoad::get()
{
  return m_pixels;
}

unsigned int ImageLoad::width()
{
  return m_width;
}

unsigned int ImageLoad::height()
{
  return m_height;
}

unsigned int ImageLoad::channels()
{
  if (m_grayscale)
    return 1;
  else
    return 3;
}

bool ImageLoad::grayscale()
{
  return m_grayscale;
}


void ImageLoad::normalise_image(float min, float max)
{
  float min_s = m_pixels[0];
  float max_s = min_s;

  for (unsigned int i = 0; i < m_pixels.size(); i++)
  {
    if (m_pixels[i] > max_s)
      max_s = m_pixels[i];

    if (m_pixels[i] < min_s)
      min_s = m_pixels[i];
  }

  float k = 0.0;
  float q = 0.0;
  if (max_s > min_s)
  {
    k = (max - min)/(max_s - min_s);
    q = max - k*max_s;
  }

  for (unsigned int i = 0; i < m_pixels.size(); i++)
    m_pixels[i] = k*m_pixels[i] + q;
}
