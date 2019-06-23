#ifndef _VECTOR_TO_IMAGE_
#define _VECTOR_TO_IMAGE_

#include <string>
#include <vector>
#include <shape.h>
#include <image_save.h>


class VectorToImage
{
    public:
        VectorToImage(std::vector<float> &vector, Shape shape);
        virtual ~VectorToImage();

        void save(std::string file_name);
        void show();

    private:
        Shape shape;
        std::vector<float> vector;

        ImageSave *image_save;
};


#endif
