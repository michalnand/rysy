#ifndef _IMAGE_LOAD_H_
#define _IMAGE_LOAD_H_

#include <vector>
#include <string>
 
class ImageLoad
{
  private:
    bool m_grayscale;
    unsigned int m_width, m_height;
    std::vector<float> m_pixels;

  public:
    ImageLoad(std::string file_name, bool load_grayscale = false, bool normalise = false);
    virtual ~ImageLoad();

  public:
    std::vector<float>& get();
    unsigned int width();
    unsigned int height();
    unsigned int channels();

    bool grayscale();

  public:
    void normalise_image(float min = 0.0, float max = 1.0);

};


#endif
