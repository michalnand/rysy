#ifndef _SHAPE_H_
#define _SHAPE_H_


struct sShape
{
    unsigned int w, h, d;
};


class Shape
{
    public:
        Shape();
        Shape(Shape& other);
        Shape(const Shape& other);

        Shape(sShape shape);
        Shape(unsigned int width, unsigned int height = 1, unsigned int depth = 1);


        virtual ~Shape();

        Shape& operator= (Shape& other);
        Shape& operator= (const Shape& other);

    protected:
        void copy(Shape& other);
        void copy(const Shape& other);

    public:
        unsigned int w();
        unsigned int h();
        unsigned int d();

        sShape get();

    public:
        void set(sShape shape);
        void set(unsigned int width, unsigned int height = 1, unsigned int depth = 1);

    public:
        unsigned int size();

    public:
        void print();

    public:
        bool operator ==(Shape &other);
        bool operator !=(Shape &other);

    private:
        sShape m_shape;
};


#endif
