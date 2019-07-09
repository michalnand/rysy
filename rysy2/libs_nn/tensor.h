#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <vector>
#include <string>
#include <config.h>

#include <shape.h>

class Tensor
{
    public:
        Tensor();
        Tensor(Tensor& other);
        Tensor(const Tensor& other);

        Tensor(unsigned int width, unsigned int height, unsigned int depth, unsigned int time = 1);
        Tensor(Shape shape);

        virtual ~Tensor();

        Tensor& operator= (Tensor& other);
        Tensor& operator= (const Tensor& other);

    protected:
        void copy(Tensor& other);
        void copy(const Tensor& other);

    public:
        unsigned int w();
        unsigned int h();
        unsigned int d();
        unsigned int t();
        unsigned int size();

        Shape shape();
        sShape shape_struct();

    public:
        void init(unsigned int width, unsigned int height, unsigned int depth, unsigned int time = 1);
        void init(Shape shape);

    public:
        void set_from_host(float *v);
        void set_from_host(std::vector<float> &v);

        void set_to_host(float *v);
        void set_to_host(std::vector<float> &v);

    public:
        void set_const(float value);
        void clear();

        void set_random(float range);

    public:
        void  set(unsigned int x, unsigned int y, unsigned int z, unsigned int t, float value);
        float get(unsigned int x, unsigned int y, unsigned int z, unsigned int t);

    public:
        void print();

        void save(std::string file_name);
        void load(std::string file_name);

        bool is_valid();

    public:

        void add(Tensor &rhs);
        void sub(Tensor &rhs);
        void mul(float value);

        void concatenate(Tensor &ta, Tensor &tb);
        void split(Tensor &ta, Tensor &tb);

        void concatenate_time_sequence(std::vector<Tensor> &source, unsigned max_time_steps = 0);
        void split_time_sequence(std::vector<Tensor> &dest);


    private:
        void init_size(Shape shape);
        unsigned int to_idx(unsigned int x, unsigned int y, unsigned int z, unsigned int t_);


    private:
        Shape m_shape;

    public:
        float *v;
};


#endif
