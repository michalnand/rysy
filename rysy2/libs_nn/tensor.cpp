#include <tensor.h>
#include <cuda_float_allocator.cuh>
#include <cuda_tensor.cuh>

#include <string.h>
#include <iostream>
#include <iomanip>
#include <fstream>

#include <math.h>

Tensor::Tensor()
{
    v = nullptr;
}

Tensor::Tensor(Tensor& other)
{
    v = nullptr;
    copy(other);
}

Tensor::Tensor(const Tensor& other)
{
    v = nullptr;
    copy(other);
}

Tensor::Tensor(unsigned int width, unsigned int height, unsigned int depth)
{
    v = nullptr;

    Shape newshape(width, height, depth);
    init(newshape);
}

Tensor::Tensor(Shape shape)
{
    v = nullptr;

    init(shape);
}

Tensor::~Tensor()
{
    if (v != nullptr)
    {
        #ifdef NETWORK_USE_CUDA
        cuda_float_allocator.free(v);
        #else
        delete[] v;
        #endif

        v = nullptr;
    }
}

Tensor& Tensor::operator= (Tensor& other)
{
    copy(other);
    return *this;
}

Tensor& Tensor::operator= (const Tensor& other)
{
    copy(other);
    return *this;
}


void Tensor::copy(Tensor& other)
{
    init_size(other.shape());

    #ifdef NETWORK_USE_CUDA
        cuda_float_allocator.device_to_device(v, other.v, size());
    #else
        memcpy(v, other.v, size()*sizeof(float));
    #endif
}

void Tensor::copy(const Tensor& other)
{
    Shape shape(other.m_shape);
    init_size(shape);

    #ifdef NETWORK_USE_CUDA
        cuda_float_allocator.device_to_device(v, other.v, size());
    #else
        memcpy(v, other.v, size()*sizeof(float));
    #endif
}


unsigned int Tensor::w()
{
    return this->m_shape.w();
}

unsigned int Tensor::h()
{
    return this->m_shape.h();
}

unsigned int Tensor::d()
{
    return this->m_shape.d();
}

unsigned int Tensor::size()
{
    return this->m_shape.size();
}

Shape Tensor::shape()
{
    return this->m_shape;
}

sShape Tensor::shape_struct()
{
    return this->m_shape.get();
}

void Tensor::init(unsigned int width, unsigned int height, unsigned int depth)
{
    Shape newshape(width, height, depth);
    init_size(newshape);
}

void Tensor::init(Shape shape)
{
    init_size(shape);
}

void Tensor::set_from_host(float *v)
{
    #ifdef NETWORK_USE_CUDA
        cuda_float_allocator.host_to_device(this->v, v, size());
    #else
        for (unsigned int i = 0; i < size(); i++)
            this->v[i] = v[i];
    #endif 
}

void Tensor::set_from_host(std::vector<float> &v)
{
    #ifdef RYSY_DEBUG

    if (v.size() != size())
    {
        std::cout << "Tensor::set_from_host inconsistent size ";
        std::cout << v.size() << " : " << size() << "\n";
        return;
    }

    #endif

    #ifdef NETWORK_USE_CUDA
        cuda_float_allocator.host_to_device(this->v, &v[0], size());
    #else
        for (unsigned int i = 0; i < size(); i++)
            this->v[i] = v[i];
    #endif
}

void Tensor::set_to_host(float *v)
{
    #ifdef NETWORK_USE_CUDA
        cuda_float_allocator.device_to_host(v, this->v, size());
    #else
        for (unsigned int i = 0; i < size(); i++)
            v[i] = this->v[i];
    #endif
}

void Tensor::set_to_host(std::vector<float> &v)
{
    #ifdef RYSY_DEBUG

    if (v.size() != size())
    {
        std::cout << "Tensor::set_to_host inconsistent size ";
        std::cout << v.size() << " : " << size() << "\n";
        return;
    }

    #endif


    #ifdef NETWORK_USE_CUDA
        cuda_float_allocator.device_to_host(&v[0], this->v, size());
    #else
        for (unsigned int i = 0; i < size(); i++)
            v[i] = this->v[i];
    #endif
}

void Tensor::set_const(float value)
{
    #ifdef NETWORK_USE_CUDA
        cuda_tensor_set_const(v, size(), value);
    #else
        for (unsigned int i = 0; i < size(); i++)
            v[i] = value;
    #endif
}

void Tensor::clear()
{
    #ifdef NETWORK_USE_CUDA
        cuda_tensor_clear(v, size());
    #else
        for (unsigned int i = 0; i < size(); i++)
            v[i] = 0.0;
    #endif
}

void Tensor::set_random(float range)
{
    #ifdef NETWORK_USE_CUDA
        cuda_tensor_random(v, size(), range);
    #else
        for (unsigned int i = 0; i < size(); i++)
            v[i] = range*((rand()%2000000)/1000000.0 - 1.0);
    #endif
}

void Tensor::set_random_xavier()
{
    float range = sqrt(2.0/size());
    set_random(range);
}

void  Tensor::set(unsigned int x, unsigned y, unsigned z, float value)
{
    unsigned int idx = to_idx(x, y, z);

    #ifdef NETWORK_USE_CUDA
        cuda_tensor_set_element(v, value, idx);
    #else
        v[idx]= value;
    #endif
}

float Tensor::get(unsigned int x, unsigned y, unsigned z)
{
    unsigned int idx = to_idx(x, y, z);

    #ifdef NETWORK_USE_CUDA
        return cuda_tensor_get_element(v, idx);
    #else
        return v[idx];
    #endif
}

void Tensor::print()
{
    std::vector<float> tmp(size());
    set_to_host(tmp);

    unsigned int ptr = 0;

    for (unsigned int d_ = 0; d_ < d(); d_++)
    {
        for (unsigned int h_ = 0; h_ < h(); h_++)
        {
            for (unsigned int w_ = 0; w_ < w(); w_++)
            {
                std::cout << std::setw(7) << std::setprecision(3) << tmp[ptr] << " ";
                ptr++;
            }
            std::cout << "\n";
        }
        std::cout << "\n";

    }

    std::cout << "\n";
}

void Tensor::save(std::string file_name)
{
    std::vector<float> tmp(size());
    set_to_host(tmp);

    std::ofstream output_file(file_name, std::ios::out | std::ios::binary);
    output_file.write( (char*)(&tmp[0]), sizeof(float)*size());
    output_file.close();
}

void Tensor::load(std::string file_name)
{
    std::vector<float> tmp(size());

    std::ifstream input_file;

    input_file.open(file_name, std::ios::in | std::ios::binary);

    if (input_file.is_open())
    {
        input_file.read( (char*)(&tmp[0]), sizeof(float)*size());
        input_file.close();

        set_from_host(tmp);
    }
}

bool Tensor::is_valid()
{
    std::vector<float> tmp(size());
    set_to_host(tmp);

    for (unsigned int i = 0; i < tmp.size(); i++)
        if (isnan(tmp[i]))
            return false;

    for (unsigned int i = 0; i < tmp.size(); i++)
        if (isinf(tmp[i]))
            return false;

    return true;
}

void Tensor::add(Tensor &rhs)
{
    #ifdef RYSY_DEBUG

    if (shape() != rhs.shape())
    {
        std::cout << "Tensor::add inconsistent tensors ";
        shape().print();
        std::cout << " : ";
        rhs.shape().print();
        std::cout << "\n";
        return;
    }

    #endif

    #ifdef NETWORK_USE_CUDA
        cuda_tensor_add(v, rhs.v, size());
    #else
        for (unsigned int i = 0; i < size(); i++)
            v[i]+= rhs.v[i];
    #endif
}

void Tensor::sub(Tensor &rhs)
{
    #ifdef RYSY_DEBUG

    if (shape() != rhs.shape())
    {
        std::cout << "Tensor::sub inconsistent tensors ";
        shape().print();
        std::cout << " : ";
        rhs.shape().print();
        std::cout << "\n";
        return;
    }

    #endif

    #ifdef NETWORK_USE_CUDA
        cuda_tensor_sub(v, rhs.v, size());
    #else
        for (unsigned int i = 0; i < size(); i++)
            v[i]-= rhs.v[i];
    #endif
}

void Tensor::mul(float value)
{
    #ifdef NETWORK_USE_CUDA
        cuda_tensor_mul(v, value, size());
    #else
        for (unsigned int i = 0; i < size(); i++)
            v[i]*= value;
    #endif
}

void Tensor::concatenate(Tensor &ta, Tensor &tb)
{
    #ifdef RYSY_DEBUG

    if (ta.size() + tb.size() != size())
    {
        std::cout << "Tensor::concatenate inconsistent size ";
        ta.shape().print();
        std::cout << " : ";
        tb.shape().print();
        std::cout << " : ";
        shape().print();
        std::cout << "\n";
        return;
    }

    #endif

    #ifdef NETWORK_USE_CUDA

        cuda_float_allocator.device_to_device(v, ta.v, ta.size());
        cuda_float_allocator.device_to_device(v + ta.size(), tb.v, tb.size());

    #else
        unsigned int ptr = 0;
        for (unsigned int i = 0; i < ta.size(); i++)
        {
            v[ptr] = ta.v[i];
            ptr++;
        }
        for (unsigned int i = 0; i < ta.size(); i++)
        {
            v[ptr] = tb.v[i];
            ptr++;
        }
    #endif
}


void Tensor::init_size(Shape shape)
{
    if (shape != m_shape)
    {
        if (v != nullptr)
        {
            #ifdef NETWORK_USE_CUDA
                cuda_float_allocator.free(v);
            #else
                delete v;
            #endif

            v = nullptr;
        }

        this->m_shape = shape;

        #ifdef NETWORK_USE_CUDA
            v = cuda_float_allocator.malloc(size());
            cuda_float_allocator.clear(v, size());
        #else
            v = new float[size()];
            for (unsigned int i = 0; i < size(); i++)
                v[i] = 0.0;
        #endif
    }
}


unsigned int Tensor::to_idx(unsigned int x, unsigned y, unsigned z)
{
    #ifdef RYSY_DEBUG

    if (x >= w() || y >= h() || z >= d())
    {
        std::cout << "Tensor::to_idx out of range";

        std::cout << x << " " << y << " " << z << " : ";
        std::cout << w()-1 << " " << h()-1 << " " << d()-1 << "\n";
        return 0;
    }

    #endif

    unsigned int result;
    result = (z*h() + y)*w() + x;
    return result;
}
