#include <tensor_to_image.h>
#include <iostream>

TensorToImage::TensorToImage()
{
    image_save = nullptr;
}

TensorToImage::TensorToImage(Shape shape)
{
    init(shape);
}

TensorToImage::~TensorToImage()
{
    if (image_save != nullptr)
        delete image_save;
}

void TensorToImage::init(Shape shape)
{
    this->shape = shape;

    v_input.resize(this->shape.size());
    for (unsigned int i = 0; i < v_input.size(); i++)
        v_input[i] = 0.0;

    make_rectangle(this->shape.d());

    output_scale  = 5;
    spacing       = 8;
    output_width  = output_scale*feature_grid_width*(this->shape.w() + spacing);
    output_height = output_scale*feature_grid_height*(this->shape.h() + spacing);

    image_save = new ImageSave(output_width, output_height, true);



    v_output.resize(output_height*output_width);
    for (unsigned int i = 0; i < output_height*output_width; i++)
        v_output[i] = 0.0;
}


void TensorToImage::save(Tensor &tensor, std::string file_name)
{
    tensor.set_to_host(v_input);
    save(v_input, file_name);
}

void TensorToImage::show(Tensor &tensor, std::string window_name)
{
    tensor.set_to_host(v_input);
    show(v_input, window_name);
}

void TensorToImage::save(std::vector<float> &vect, std::string file_name)
{
    process(v_output, vect);
    normalise(v_output);
    image_save->save(file_name, v_output);
}

void TensorToImage::show(std::vector<float> &vect, std::string window_name)
{
    process(v_output, vect);
    normalise(v_output);
    image_save->show(v_output, window_name);
}




void TensorToImage::make_rectangle(unsigned int features_count)
{
    unsigned int width = sqrt(features_count);

    while ((features_count%width) != 0)
        width++;

    feature_grid_width  = width;
    feature_grid_height = features_count/width;
}

std::vector<std::vector<float>> TensorToImage::extract_feature_map(std::vector<float> &vect, unsigned int map_idx)
{
    std::vector<std::vector<float>> result;

    result.resize(shape.h());
    for (unsigned int j = 0; j < shape.h(); j++)
    {
        result[j].resize(shape.w());
    }

    for (unsigned int j = 0; j < shape.h(); j++)
        for (unsigned int i = 0; i < shape.w(); i++)
        {
            result[j][i] = vect[(map_idx*shape.h() + j)*shape.w() + i];
        }

    return result;
}

void TensorToImage::process(std::vector<float> &v_output, std::vector<float> &v_input)
{
    for (unsigned int f = 0; f < shape.d(); f++)
    {
        auto feature_map = extract_feature_map(v_input, f);

        unsigned int x_offset = (f%feature_grid_width)*(shape.w() + spacing) + spacing/2;
        unsigned int y_offset = (f/feature_grid_width)*(shape.h() + spacing) + spacing/2;

        for (unsigned int j = 0; j < shape.h(); j++)
            for (unsigned int i = 0; i < shape.w(); i++)
            {
                for (unsigned int y = 0; y < output_scale; y++)
                    for (unsigned int x = 0; x < output_scale; x++)
                    {
                        unsigned int output_idx = ((j + y_offset)*output_scale + y)*output_width + (i + x_offset)*output_scale + x;
                        v_output[output_idx] = feature_map[j][i];
                    }
            }
    }
}

void TensorToImage::normalise(std::vector<float> &vect)
{
    float max = vect[0];
    float min = max;

    for (unsigned int j = 0; j < vect.size(); j++)
    {
        if (vect[j] > max)
            max = vect[j];

        if (vect[j] < min)
            min = vect[j];
    }

    float k = 0.0;
    float q = 0.0;

    if (max > min)
    {
        k = (1.0 - 0.0)/(max - min);
        q = 1.0 - k*max;
    }

    for (unsigned int j = 0; j < vect.size(); j++)
        vect[j] = k*vect[j] + q;
}
