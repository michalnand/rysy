#ifndef _IMAGE_SAVE_H_
#define _IMAGE_SAVE_H_

#include <vector>
#include <string>
#include <CImg.h>

class ImageSave
{
    private:
        bool m_grayscale, m_display_enabled;
        unsigned int m_width, m_height;

    private:
        cimg_library::CImg<float> *output_image;
        cimg_library::CImgDisplay *image_display;

    public:
        ImageSave(unsigned int width, unsigned int height, bool grayscale, bool display_enabled = false);
        virtual ~ImageSave();

        void save(std::string file_name, std::vector<float> &v);
        void show(std::vector<float> &v, std::string window_name = "");

    private:
        void vector_to_image(std::vector<float> v);
        void normalise(std::vector<float> &v, float min, float max);

};


#endif
