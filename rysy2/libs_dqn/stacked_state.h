#ifndef _STACKED_STATE_H_
#define _STACKED_STATE_H_

#include <shape.h>
#include <vector>

class StackedState
{
    public:
        StackedState();
        StackedState(Shape input_shape, unsigned int frames);
        StackedState(StackedState& other);
        StackedState(const StackedState& other);

        virtual ~StackedState();
        StackedState& operator= (StackedState& other);
        StackedState& operator= (const StackedState& other);

    protected:
        void copy(StackedState& other);
        void copy(const StackedState& other);

    public:
        void init(Shape input_shape, unsigned int frames);

        Shape& shape();
        std::vector<float>& get();

        void set(unsigned int x, unsigned int y, unsigned int z, float v);
        float get(unsigned int x, unsigned int y, unsigned int z, unsigned int frame);

        void next_frame();
        void clear();
        void print();

        void random();

    protected:
        Shape input_shape, output_shape;
        unsigned int frames;
        std::vector<float> state;
};

#endif
