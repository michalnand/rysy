#include <weights.h>
#include <kernels/solver_adam.cuh>


Weights::Weights()
{

}

Weights::Weights(Weights& other)
{
    copy(other);
}

Weights::Weights(const Weights& other)
{
    copy(other);
}

Weights::Weights(unsigned int width, unsigned int height, unsigned int depth, unsigned int time)
{
    init(width, height, depth, time);
}

Weights::Weights(Shape shape)
{
    init(shape);
}

Weights::~Weights()
{

}

Weights& Weights::operator= (Weights& other)
{
    copy(other);
    return *this;
}

Weights& Weights::operator= (const Weights& other)
{
    copy(other);
    return *this;
}


void Weights::copy(Weights& other)
{
    this->m         = other.m;
    this->v         = other.v;
    this->weights   = other.weights;
    this->gradient    = other.gradient;
}

void Weights::copy(const Weights& other)
{
    this->m         = other.m;
    this->v         = other.v;
    this->weights   = other.weights;
    this->gradient    = other.gradient;
}


unsigned int Weights::w()
{
    return weights.w();
}

unsigned int Weights::h()
{
    return weights.h();
}

unsigned int Weights::d()
{
    return weights.d();
}

unsigned int Weights::t()
{
    return weights.t();
}

unsigned int Weights::size()
{
    return weights.size();
}

Shape Weights::shape()
{
    return weights.shape();
}

sShape Weights::shape_struct()
{
    return weights.shape().get();
}


void Weights::init(unsigned int width, unsigned int height, unsigned int depth, unsigned int time)
{
    Shape newshape(width, height, depth, time);
    init_size(newshape);
}

void Weights::init(Shape newshape)
{
    init_size(newshape);
}

void Weights::train(float learning_rate, float lambda1, float lambda2, float gradient_clip)
{
    solver_adam(weights, gradient, m, v, learning_rate, lambda1, lambda2, gradient_clip);
    gradient.clear();
}

void Weights::init_size(Shape shape)
{
    weights.init(shape);

    gradient.init(shape);
    m.init(shape);
    v.init(shape);

    gradient.clear();
    m.clear();
    v.clear();
}
