#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <CImg.h>
#include <string>
#include <vector>

struct sPixel
{
  std::vector<float> b;
};

class Image
{
  private:
    cimg_library::CImg<float> *c_img;
    cimg_library::CImgDisplay *image_display;

    unsigned int current_row;
  public:
    std::vector<std::vector<struct sPixel>> pixels;
    unsigned int width, height;



  public:
    //load image from file
    Image(std::string file_name, bool grayscale = false);

    //create empty image
    Image(unsigned int width, unsigned int height);
    ~Image();

    void save(std::string file_name);

    void show();

    std::vector<float> as_vector(bool grayscale = false);

    void from_vector(std::vector<float> &v_image);
    void from_vector(float *v_image);

    void from_vector_grayscale(std::vector<float> &v_image);
    void from_vector_grayscale(float *v_image);

    void from_feature_map(std::vector<float> &v_image, unsigned int maps_count);
    void from_feature_map(float *v_image, unsigned int maps_count);

    void add_row(float *row);

    float compare(class Image &image);
};

#endif
