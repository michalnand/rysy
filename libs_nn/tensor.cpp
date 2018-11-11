#include "tensor.h"
#include "cuda_float_allocator.cuh"
#include "cuda_tensor.cuh"

#include <string.h>
#include <fstream>
#include <math.h>

Tensor::Tensor()
{
  m_w = 0;
  m_h = 0;
  m_d = 0;
  m_size = 0;
  v = nullptr;
}

Tensor::Tensor(Tensor& other)
{
  m_w = 0;
  m_h = 0;
  m_d = 0;
  m_size = 0;
  v = nullptr;

  copy(other);
}

Tensor::Tensor(const Tensor& other)
{
  m_w = 0;
  m_h = 0;
  m_d = 0;
  m_size = 0;
  v = nullptr;

  copy(other);
}

Tensor::~Tensor()
{
  if (v != nullptr)
  {
    #ifdef NETWORK_USE_CUDA
      cu_free(v);
    #else
      delete v;
    #endif

    v = nullptr;
  }

  m_w = 0;
  m_h = 0;
  m_d = 0;
  m_size = 0;
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


Tensor::Tensor(sGeometry geometry)
{
  m_w = 0;
  m_h = 0;
  m_d = 0;
  m_size = 0;
  v = nullptr;

  init(geometry.w, geometry.h, geometry.d);
}

Tensor::Tensor(unsigned int w_, unsigned int h_, unsigned int d_)
{
  m_w = 0;
  m_h = 0;
  m_d = 0;
  m_size = 0;
  v = nullptr;

  init(w_, h_, d_);
}

unsigned int Tensor::size()
{
  return m_size;
}

unsigned int Tensor::w()
{
  return m_w;
}

unsigned int Tensor::h()
{
  return m_h;
}

unsigned int Tensor::d()
{
  return m_d;
}



void Tensor::set_from_host(float *v)
{
  #ifdef NETWORK_USE_CUDA
  cu_host_to_device(this->v, v, size());
  #else
  for (unsigned int i = 0; i < size(); i++)
    this->v[i] = v[i];
  #endif
}

void Tensor::set_from_host(std::vector<float> &v)
{
  set_from_host(&v[0]);
}

void Tensor::set_to_host(float *v)
{
  #ifdef NETWORK_USE_CUDA
  cu_device_to_host(v, this->v, size());
  #else
  for (unsigned int i = 0; i < size(); i++)
    v[i] = this->v[i];
  #endif
}

void Tensor::set_to_host(std::vector<float> &v)
{
  set_to_host(&v[0]);
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

void Tensor::add(Tensor &rhs)
{
  #ifdef NETWORK_USE_CUDA
    cuda_tensor_add(v, rhs.v, size());
  #else
    for (unsigned int i = 0; i < size(); i++)
      v[i]+= rhs.v[i];
  #endif
}

void Tensor::sub(Tensor &rhs)
{
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

void Tensor::regularization_l1(float lambda)
{
  #ifdef NETWORK_USE_CUDA
    cuda_tensor_regularization_l1(v, lambda, size());
  #else
    for (unsigned int i = 0; i < size(); i++)
    {
      if (v[i] > 0.0)
        v[i]-= lambda;
      else
        v[i]+= lambda;
    }
  #endif
}

void Tensor::regularization_l2(float lambda)
{
  #ifdef NETWORK_USE_CUDA
    cuda_tensor_regularization_l2(v, lambda, size());
  #else
    for (unsigned int i = 0; i < size(); i++)
    {
      v[i]-= lambda*v[i];
    }
  #endif
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


sGeometry Tensor::get_geometry()
{
  sGeometry result;
  result.w = w();
  result.h = h();
  result.d = d();

  return result;
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
        printf("%6.3f ", tmp[ptr]);
        ptr++;
      }
      printf("\n");
    }

    printf("\n");
  }

  printf("\n");
}


void Tensor::save_to_file(std::string file_name)
{
  std::vector<float> tmp(size());
  set_to_host(tmp);

  std::ofstream output_file(file_name, std::ios::out | std::ios::binary);
  output_file.write( (char*)(&tmp[0]), sizeof(float)*size());
  output_file.close();
}

void Tensor::load_from_file(std::string file_name)
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


void Tensor::copy(Tensor &other)
{
  if (other.m_size != m_size)
  {
    if (v != nullptr)
    {
      #ifdef NETWORK_USE_CUDA
        cu_free(v);
      #else
        delete v;
      #endif

      v = nullptr;
    }

    #ifdef NETWORK_USE_CUDA
      cu_device_to_device(v, other.v, other.m_size);
    #else
      v = new float[other.m_size];
    #endif
  }

  m_w = other.m_w;
  m_h = other.m_h;
  m_d = other.m_d;

  m_size = m_w*m_h*m_d;

  #ifdef NETWORK_USE_CUDA
    cu_device_to_device(v, other.v, size());
  #else
    memcpy(v, other.v, size()*sizeof(float));
  #endif
}

void Tensor::copy(const Tensor &other)
{
  if ( (other.m_w != m_w)||(other.m_h != m_h)||(other.m_d != m_d))
  {
    if (v != nullptr)
    {
      #ifdef NETWORK_USE_CUDA
        cu_free(v);
      #else
        delete v;
      #endif

      v = nullptr;
    }
  }

  m_w = other.m_w;
  m_h = other.m_h;
  m_d = other.m_d;

  m_size = m_w*m_h*m_d;

  #ifdef NETWORK_USE_CUDA
    v = cu_malloc(size());
    cu_device_to_device(v, other.v, size());
  #else
    v = new float[size()];
    memcpy(v, other.v, size()*sizeof(float));
  #endif
}

unsigned int Tensor::to_idx(unsigned int x, unsigned y, unsigned z)
{
  unsigned int result;
  result = (z*h() + y)*w() + x;
  return result;
}


void Tensor::init(sGeometry geometry)
{
  init(geometry.w, geometry.h, geometry.d);
}


void Tensor::init(unsigned int w_, unsigned h_, unsigned d_)
{
  if (v != nullptr)
  {
    #ifdef NETWORK_USE_CUDA
      cu_free(v);
    #else
      delete v;
    #endif

    v = nullptr;
  }


  m_w = w_;
  m_h = h_;
  m_d = d_;

  m_size = m_w*m_h*m_d;

  #ifdef NETWORK_USE_CUDA
    v = cu_malloc(size());
  #else
    v = new float[size()];
  #endif

  clear();
}

float Tensor::rms(Tensor &rhs)
{
  float result;

  #ifdef NETWORK_USE_CUDA
    cuda_rms(&result, v, rhs.v, size());
  #else
    result = 0.0;
    for (unsigned int i = 0; i < size(); i++)
    {
      float tmp = v[i] - rhs.v[i];
      result+= tmp*tmp;
    }

    result = sqrt(result);
  #endif

  return result;
}
