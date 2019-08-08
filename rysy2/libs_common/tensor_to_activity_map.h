#ifndef _TENSOR_TO_ACTIVITY_MAP_H_
#define _TENSOR_TO_ACTIVITY_MAP_H_

#include <string>
#include <vector>
#include <shape.h>
#include <tensor.h>
#include <image_save.h>

class TensorToActivityMap
{
    public:
        TensorToActivityMap();
        TensorToActivityMap(Shape shape);
        virtual ~TensorToActivityMap();

        TensorToActivityMap(TensorToActivityMap &other);
        TensorToActivityMap(const TensorToActivityMap &other);

        TensorToActivityMap& operator=(TensorToActivityMap &rhs);
        TensorToActivityMap& operator=(const TensorToActivityMap &rhs);

    private:
        void copy(TensorToActivityMap &other);
        void copy(const TensorToActivityMap &other);

    public:
        void init(Shape shape);
        void clear();
        void add(Tensor &v);

        void save(std::string output_path, unsigned int output_scale = 1);

        Shape shape();

    private:
        void compute_result();
        std::vector<float> scale(unsigned int output_scale, bool max_color);
        std::vector<std::vector<float>> create_color_map(unsigned int count);

        std::vector<float> get_max_color(unsigned int x, unsigned int y);
        std::vector<float> get_average_color(unsigned int x, unsigned int y);

    private:
        Shape m_shape;
        Tensor t_sum;
        std::vector<float> v_sum;
        std::vector<std::vector<std::vector<float>>> result;

        std::vector<std::vector<float>> m_color_map;
};

#endif
