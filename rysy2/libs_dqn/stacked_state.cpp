#include <stacked_state.h>
#include <iostream>

StackedState::StackedState()
{

}

StackedState::StackedState(Shape input_shape, unsigned int frames)
{
    init(input_shape, frames);
}

StackedState::StackedState(StackedState& other)
{
    copy(other);
}

StackedState::StackedState(const StackedState& other)
{
    copy(other);
}

StackedState::~StackedState()
{

}

StackedState& StackedState::operator= (StackedState& other)
{
    copy(other);
    return *this;
}

StackedState& StackedState::operator= (const StackedState& other)
{
    copy(other);
    return *this;
}

void StackedState::copy(StackedState& other)
{
    input_shape   = other.input_shape;
    output_shape   = other.output_shape;
    frames  = other.frames;
    state   = other.state;
}

void StackedState::copy(const StackedState& other)
{
    input_shape   = other.input_shape;
    output_shape   = other.output_shape;
    frames  = other.frames;
    state   = other.state;
}

void StackedState::init(Shape input_shape, unsigned int frames)
{
    this->input_shape     = input_shape;
    this->frames          = frames;

    this->output_shape.set(this->input_shape.w(), this->input_shape.h(), this->input_shape.d()*this->frames);

    this->state.resize(this->output_shape.size());
}

Shape& StackedState::shape()
{
    return output_shape;
}

std::vector<float>& StackedState::get()
{
    return state;
}

void StackedState::set(unsigned int x, unsigned int y, unsigned int z, float v)
{
    unsigned int idx = (z*input_shape.h() + y)*input_shape.w() + x;
    state[idx] = v;
}

float StackedState::get(unsigned int x, unsigned int y, unsigned int z, unsigned int frame)
{
    unsigned int idx = ((frame*input_shape.d() + z)*input_shape.h() + y)*input_shape.w() + x;
    return state[idx];
}

void StackedState::next_frame()
{
    for (int frame = frames-1; frame > 0; frame--)
        for (unsigned int i = 0; i < input_shape.size(); i++)
            state[i + frame*input_shape.size()] = state[i + (frame-1)*input_shape.size()];

    for (unsigned int i = 0; i < input_shape.size(); i++)
        state[i] = 0.0;
}

void StackedState::clear()
{
    for (unsigned int i = 0; i < state.size(); i++)
        state[i] = 0.0;
}

void StackedState::print()
{
    for (unsigned int frame = 0; frame < frames; frame++)
    {
        for (unsigned int z = 0; z < input_shape.d(); z++)
        {
            for (unsigned int y = 0; y < input_shape.h(); y++)
            {
                for (unsigned int x = 0; x < input_shape.w(); x++)
                    std::cout << get(x, y, z, frame) << " ";

                std::cout << "\n";
            }
            std::cout << "\n";
        }
        std::cout << "\n\n\n";
    }

    for (unsigned int i = 0; i < 32; i++)
        std::cout << "=";

    std::cout << "\n\n\n";
}

void StackedState::random()
{
    for (unsigned int i = 0; i < state.size(); i++)
        state[i] = (rand()%100000)/100000.0;
}
