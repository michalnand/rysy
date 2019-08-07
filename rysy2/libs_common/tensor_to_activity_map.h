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
        std::vector<float> scale(unsigned int output_scale);

    private:
        Shape m_shape;
        Tensor t_sum;
        std::vector<float> v_sum;
        std::vector<float> result;
};

#endif
