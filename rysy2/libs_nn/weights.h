#ifndef _WEIGHTS_H_
#define _WEIGHTS_H_

#include <tensor.h>

class Weights
{
    public:
        Weights();
        Weights(Weights& other);
        Weights(const Weights& other);

        Weights(unsigned int width, unsigned int height, unsigned int depth, unsigned int time = 1);
        Weights(Shape shape);

        virtual ~Weights();

        Weights& operator= (Weights& other);
        Weights& operator= (const Weights& other);

    protected:
        void copy(Weights& other);
        void copy(const Weights& other);

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

        void train(float learning_rate, float lambda1, float lambda2, float gradient_clip);

    private:
        void init_size(Shape shape);

    private:
        Tensor m, v;

    public:
        Tensor weights, gradient;
};

#endif
