#include <vector_to_image.h>


VectorToImage::VectorToImage(std::vector<float> &vector, Shape shape)
{
    this->vector = vector;
    this->shape  = shape;

    bool grayscale;
    if (this->shape.d() == 1)
        grayscale = true;
    else
        grayscale = false;

    image_save = new ImageSave(this->shape.w(), this->shape.h(), grayscale);
}

VectorToImage::~VectorToImage()
{
    delete image_save;
}

void VectorToImage::save(std::string file_name)
{
    image_save->save(file_name, vector);
}

void VectorToImage::show()
{
    image_save->show(vector);
}
