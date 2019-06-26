#ifndef _TENSOR_TO_IMAGE_
#define _TENSOR_TO_IMAGE_

#include <string>
#include <vector>
#include <shape.h>
#include <tensor.h>
#include <image_save.h>


class TensorToImage
{
    public:
        TensorToImage();
        TensorToImage(Shape shape);
        virtual ~TensorToImage();

        void init(Shape shape);

        void save(Tensor &tensor, std::string file_name);
        void show(Tensor &tensor, std::string window_name = "");

        void save(std::vector<float> &vect, std::string file_name);
        void show(std::vector<float> &vect, std::string window_name = "");

    private:
        void make_rectangle(unsigned int features_count);
        std::vector<std::vector<float>> extract_feature_map(std::vector<float> &vect, unsigned int map_idx);
        void process(std::vector<float> &v_output, std::vector<float> &v_input);

        void normalise(std::vector<float> &vect);
    private:
        Shape shape;
        std::vector<float> v_input, v_output;

        unsigned int feature_grid_width, feature_grid_height;
        unsigned int output_scale, spacing;
        unsigned int output_width, output_height;

        ImageSave *image_save;
};


#endif
