#include "image.h"
#include <stdio.h>

Image::Image(std::string file_name, bool grayscale)
{
  this->height = 0;
  this->width = 0;
  c_img = new cimg_library::CImg<float>(file_name.c_str());

  if (c_img == nullptr)
  {
    printf("Image %s opening ERROR\n", file_name.c_str());
    return;
  }
  else
  {
  //  printf("Image %s [%u %u]\n", file_name.c_str(), (unsigned int)c_img->height(), (unsigned int)c_img->width());
  }

  c_img->normalize(0.0, 1.0); //normalise values into <-1, 1> interval

  for (unsigned int j = 0; j < (unsigned int)c_img->height(); j++)
  {
    std::vector<struct sPixel> tmp;
    struct sPixel pixel;
    pixel.b.push_back(0.0);
    pixel.b.push_back(0.0);
    pixel.b.push_back(0.0);

    for (unsigned int i = 0; i < (unsigned int)c_img->width(); i++)
    {
      if (grayscale)
      {
        float tmp = 0.0;
        tmp+= (*(c_img->data(i, j, 0, 0)));
        tmp+= (*(c_img->data(i, j, 0, 1)));
        tmp+= (*(c_img->data(i, j, 0, 2)));

        tmp/= 3.0;
        pixel.b[0] = tmp;
        pixel.b[1] = tmp;
        pixel.b[2] = tmp;
      }
      else
      {
        pixel.b[0] = (*(c_img->data(i, j, 0, 0)));
        pixel.b[1] = (*(c_img->data(i, j, 0, 1)));
        pixel.b[2] = (*(c_img->data(i, j, 0, 2)));
      }
      tmp.push_back(pixel);
    }


    pixels.push_back(tmp);
  }

  this->height = c_img->height();
  this->width = c_img->width();

  image_display = nullptr;

  current_row = 0;
}

Image::Image(unsigned int width, unsigned int height)
{
  std::vector<struct sPixel> row;

  struct sPixel pixel;
  pixel.b.push_back(0.0);
  pixel.b.push_back(0.0);
  pixel.b.push_back(0.0);

  for (unsigned int i = 0; i < width; i++)
  {
    row.push_back(pixel);
  }


  for (unsigned int j = 0; j < height; j++)
  {
    pixels.push_back(row);
  }

  c_img = new cimg_library::CImg<float>(pixels[0].size(), pixels.size(), 1, 3, 0);
  image_display = nullptr;

  this->height = height;
  this->width = width;

  current_row = 0;
}


Image::~Image()
{
  if (image_display != nullptr)
  {
    delete image_display;
    image_display = nullptr;
  }

  if (c_img != nullptr)
  {
    delete c_img;
    c_img = nullptr;
  }
}


void Image::save(std::string file_name)
{
  for (unsigned int j = 0; j < pixels.size(); j++)
    for (unsigned int i = 0; i < pixels[j].size(); i++)
    {
      c_img->draw_point(i, j, &pixels[j][i].b[0]);
    }

  c_img->normalize(0, 255);
  c_img->save(file_name.c_str());
}

void Image::show()
{
  if (image_display != nullptr)
  {
    delete image_display;
    image_display = nullptr;
  }


  for (unsigned int j = 0; j < pixels.size(); j++)
    for (unsigned int i = 0; i < pixels[j].size(); i++)
    {
      c_img->draw_point(i, j, &pixels[j][i].b[0]);
    }

  char name[1024];
  sprintf(name, "image_%u", (unsigned int)((unsigned long int)this));

  c_img->normalize(0, 255);

  image_display = new cimg_library::CImgDisplay(*c_img, name);
}

std::vector<float> Image::as_vector(bool grayscale)
{
  std::vector<float> result;

  if (grayscale == true)
  {
    for (unsigned int j = 0; j < pixels.size(); j++)
      for (unsigned int i = 0; i < pixels[j].size(); i++)
        result.push_back((pixels[j][i].b[0] + pixels[j][i].b[1] + pixels[j][i].b[2])/3.0);
  }
  else
  {
    for (unsigned int ch = 0; ch < 3; ch++)
      for (unsigned int j = 0; j < pixels.size(); j++)
        for (unsigned int i = 0; i < pixels[j].size(); i++)
            result.push_back(pixels[j][i].b[ch]);
  }


  return result;
}


void Image::from_vector(std::vector<float> &v_image)
{
  from_vector(&v_image[0]);
}


void Image::from_vector(float *v_image)
{
  unsigned int ptr = 0;

  for (unsigned int ch = 0; ch < 3; ch++)
    for (unsigned int j = 0; j < pixels.size(); j++)
      for (unsigned int i = 0; i < pixels[j].size(); i++)
        {
          float tmp =  v_image[ptr];
          ptr++;


          if (tmp > 1.0)
            tmp = 1.0;

          if (tmp < 0.0)
            tmp = 0.0;

          pixels[j][i].b[ch] = tmp;
        }
}

void Image::from_vector_grayscale(std::vector<float> &v_image)
{
  from_vector_grayscale(&v_image[0]);
}

void Image::from_vector_grayscale(float *v_image)
{
  from_feature_map(v_image, 1);
}

void Image::from_feature_map(std::vector<float> &v_image, unsigned int maps_count)
{
  from_feature_map(&v_image[0], maps_count);
}

void Image::from_feature_map(float *v_image, unsigned int maps_count)
{
  unsigned int map_height = height/maps_count;


  for (unsigned int map = 0; map < maps_count; map++)
  {
    float max = -1000000000.0;
    float min = -max;

    for (unsigned int j = 0; j < map_height; j++)
      for (unsigned int i = 0; i < width; i++)
        {
          unsigned int idx = (j*width + i)*maps_count + map;
          float tmp = v_image[idx];

          if (tmp < 0.0)
            tmp = 0.0;

          if (tmp > max)
            max = tmp;

          if (tmp < min)
            min = tmp;

          for (unsigned int ch = 0; ch < 3; ch++)
            pixels[map*map_height + j][i].b[ch] = tmp;
        }

        float k = 0.0;
        float q = 0.0;

        if (max > min)
        {
          k = (1.0 - 0.0)/(max - min);
          q = 1.0 - k*max;
        }

        for (unsigned int j = 0; j < map_height; j++)
          for (unsigned int i = 0; i < width; i++)
            for (unsigned int ch = 0; ch < 3; ch++)
              pixels[map*map_height + j][i].b[ch] = k*pixels[map*map_height + j][i].b[ch] + q;
  }

}


void Image::add_row(float *row)
{
  for (unsigned int i = 0; i < pixels[current_row].size(); i++)
    for (unsigned int ch = 0; ch < 3; ch++)
      pixels[current_row][i].b[ch] = row[i];

      /*
  if ((current_row+1) < pixels.size())
    current_row++;
    */

  current_row = (current_row+1)%pixels.size();
}

float Image::compare(class Image &image)
{
  float result = 0.0;

  unsigned int channels = 3;

  for (unsigned int j = 0; j < pixels.size(); j++)
    for (unsigned int i = 0; i < pixels[j].size(); i++)
      for (unsigned int ch = 0; ch < channels; ch++)
      {
        float tmp = this->pixels[j][i].b[ch] - image.pixels[j][i].b[ch];
        result+= tmp*tmp;
      }

  result = sqrt(result/(pixels.size()*pixels[0].size()*channels));
  return result;
}
