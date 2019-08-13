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

Tensor::Tensor(unsigned int width, unsigned int height, unsigned int depth, unsigned int time)
{
    v = nullptr;

    Shape newshape(width, height, depth, time);
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

unsigned int Tensor::t()
{
    return this->m_shape.t();
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

void Tensor::init(unsigned int width, unsigned int height, unsigned int depth, unsigned int time)
{
    Shape newshape(width, height, depth, time);
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
        std::cout << v.size() << " : expecting " << size() << "\n";
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
        std::cout << v.size() << " : expecting " << size() << "\n";
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

void  Tensor::set(unsigned int x, unsigned y, unsigned z, unsigned int t, float value)
{
    unsigned int idx = to_idx(x, y, z, t);

    #ifdef NETWORK_USE_CUDA
        cuda_tensor_set_element(v, value, idx);
    #else
        v[idx]= value;
    #endif
}

float Tensor::get(unsigned int x, unsigned int y, unsigned int z, unsigned int t)
{
    unsigned int idx = to_idx(x, y, z, t);

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

    for (unsigned int t_ = 0; t_ < t(); t_++)
    {
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
    else
    {
        std::cout << "Tensor::load file opening error : " << file_name << "\n";
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
        std::cout << " : expecting ";
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
        std::cout << " : expecting ";
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
        std::cout << " : expecting ";
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



void Tensor::split(Tensor &ta, Tensor &tb)
{
    #ifdef RYSY_DEBUG

    if (ta.size() + tb.size() != size())
    {
        std::cout << "Tensor::split inconsistent size ";
        ta.shape().print();
        std::cout << " : ";
        tb.shape().print();
        std::cout << " : expecting ";
        shape().print();
        std::cout << "\n";
        return;
    }

    #endif

    #ifdef NETWORK_USE_CUDA

        cuda_float_allocator.device_to_device(ta.v, v, ta.size());
        cuda_float_allocator.device_to_device(tb.v, v + ta.size(), tb.size());

    #else
        unsigned int ptr = 0;
        for (unsigned int i = 0; i < ta.size(); i++)
        {
            va[i] = v[ptr];
            ptr++;
        }
        for (unsigned int i = 0; i < tb.size(); i++)
        {
            vb[i] = v[ptr];
            ptr++;
        }
    #endif
}


void Tensor::concatenate_time_sequence(std::vector<Tensor> &source, unsigned int max_time_steps)
{
    #ifdef RYSY_DEBUG

    if (max_time_steps == 0 && source.size() != t())
    {
        std::cout << "Tensor::concatenate_time_sequence inconsistent time sequence length :";
        std::cout << source.size() << " expecting : " << t() << "\n";
        return;
    }
 
    for (unsigned int i = 0; i < source.size(); i++)
    {
        if ((source[i].w() != w()) || (source[i].h() != h()) || (source[i].d() != d()))
        {
            std::cout << "Tensor::concatenate_time_sequence inconsistent shape :";
            std::cout << source[i].shape().w() << " " << source[i].shape().h() << " " << source[i].shape().d() << " " << " expecting : " << w() << " " << h() << " " << d() << "\n";
            return;
        }
    }

    #endif

    //concatenate source tensors into v
    unsigned int offset = 0;
    for (unsigned int i = 0; i < source.size(); i++)
    {
        #ifdef NETWORK_USE_CUDA

            cuda_float_allocator.device_to_device(v + offset, source[i].v, source[i].size());

        #else
            unsigned int ptr = offset;
            for (unsigned int i = 0; i < source[i].size(); i++)
            {
                v[ptr] = dest[i].v;
                ptr++;
            }
        #endif

        offset+= source[i].size();
    }
}

void Tensor::split_time_sequence(std::vector<Tensor> &dest)
{
    //re-init length, if incosistent
    if (dest.size() != t())
    {
        dest.resize(t());
    }

    //re-init tensors if shape doesn't match
    Shape target_shape(w(), h(), d());
    for (unsigned int i = 0; i < dest.size(); i++)
        if (dest[i].shape() != target_shape)
        {
            dest[i].init(target_shape);
        }

    //split tensor into dest
    unsigned int offset = 0;
    for (unsigned int i = 0; i < dest.size(); i++)
    {
        #ifdef NETWORK_USE_CUDA

            cuda_float_allocator.device_to_device(dest[i].v, v + offset, target_shape.size());

        #else
            unsigned int ptr = offset;
            for (unsigned int i = 0; i < target_shape.size(); i++)
            {
                dest[i].v = v[ptr];
                ptr++;
            }
        #endif

        offset+= target_shape.size();
    }
}

float Tensor::norm_l2()
{
    std::vector<float> v_tmp(size());
    set_to_host(v_tmp);

    float result = 0.0;

    for (unsigned int i = 0; i < size(); i++)
    {
        float tmp = v_tmp[i];
        result+= tmp*tmp;
    }

    return result;
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


unsigned int Tensor::to_idx(unsigned int x, unsigned int y, unsigned int z, unsigned int t_)
{
    #ifdef RYSY_DEBUG

    if (x >= w() || y >= h() || z >= d() || t_ >= t())
    {
        std::cout << "Tensor::to_idx out of range";

        std::cout << x << " " << y << " " << z << " : " << t_ << " : ";
        std::cout << w()-1 << " " << h()-1 << " " << d()-1 << t()-1 << "\n";
        return 0;
    }

    #endif

    unsigned int result;
    result = ((t_*d() + z)*h() + y)*w() + x;
    return result;
}
