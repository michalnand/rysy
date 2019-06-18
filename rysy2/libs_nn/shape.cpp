#include <shape.h>
#include <iostream>

Shape::Shape()
{
    this->m_shape.w = 0;
    this->m_shape.h = 0;
    this->m_shape.d = 0;
}

Shape::Shape(Shape& other)
{
    copy(other);
}

Shape::Shape(const Shape& other)
{
    copy(other);
}

Shape::Shape(sShape shape)
{
    set(shape);
}

Shape::Shape(unsigned int width, unsigned int height, unsigned int depth)
{
    set(width, height, depth);
}


Shape::~Shape()
{

}

Shape& Shape::operator= (Shape& other)
{
    copy(other);
    return *this;
}

Shape& Shape::operator= (const Shape& other)
{
    copy(other);
    return *this;
}

Shape& Shape::operator= (int other[])
{
    set(other[0], other[1], other[2]);
    return *this;
}

Shape& Shape::operator= (const int other[])
{
    set(other[0], other[1], other[2]);
    return *this;
}

Shape& Shape::operator= (unsigned int other[])
{
    set(other[0], other[1], other[2]);
    return *this;
}

Shape& Shape::operator= (const unsigned int other[])
{
    set(other[0], other[1], other[2]);
    return *this;
}



void Shape::copy(Shape& other)
{
    this->m_shape = other.m_shape;
}

void Shape::copy(const Shape& other)
{
    this->m_shape = other.m_shape;
}

unsigned int Shape::w()
{
    return this->m_shape.w;
}

unsigned int Shape::h()
{
    return this->m_shape.h;
}

unsigned int Shape::d()
{
    return this->m_shape.d;
}

sShape Shape::get()
{
    return this->m_shape;
}


void Shape::set(sShape shape)
{
    this->m_shape = shape;
}

void Shape::set(unsigned int width, unsigned int height, unsigned int depth)
{
    this->m_shape.w = width;
    this->m_shape.h = height;
    this->m_shape.d = depth;
}


unsigned int Shape::size()
{
    return this->m_shape.w*this->m_shape.h*this->m_shape.d;
}

void Shape::print()
{
    std::cout << w() << " ";
    std::cout << h() << " ";
    std::cout << d() << " ";
}

bool Shape::operator ==(Shape &other)
{
    if (w() != other.w())
        return false;

    if (h() != other.h())
        return false;

    if (d() != other.d())
        return false;

    return true;
}

bool Shape::operator !=(Shape &other)
{
    if (w() != other.w())
        return true;

    if (h() != other.h())
        return true;

    if (d() != other.d())
        return true;

    return false;
}


bool Shape::operator ==(const Shape &other)
{
    if (w() != other.m_shape.w)
        return false;

    if (h() != other.m_shape.h)
        return false;

    if (d() != other.m_shape.d)
        return false;

    return true;
}

bool Shape::operator !=(const Shape &other)
{
    if (w() != other.m_shape.w)
        return true;

    if (h() != other.m_shape.h)
        return true;

    if (d() != other.m_shape.d)
        return true;

    return false;
}
