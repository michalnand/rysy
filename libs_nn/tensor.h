#ifndef _TENSOR_H_
#define _TENSOR_H_

#include "nn_struct.h"
#include "network_config.h"

#include <vector>
#include <string>

class Tensor
{
  public:
    // Default constructor
    Tensor();

    // Copy constructor
    Tensor(Tensor& other);

    // Copy constructor
    Tensor(const Tensor& other);

    // Destructor
    virtual ~Tensor();

    // Copy assignment operator
    Tensor& operator= (Tensor& other);

    // Copy assignment operator
    Tensor& operator= (const Tensor& other);

  public:
    Tensor(sGeometry geometry);
    Tensor(unsigned int w_, unsigned int h_ = 1, unsigned int d_ = 1);

  public:
    unsigned int size();
    unsigned int w();
    unsigned int h();
    unsigned int d();

  public:
    void set_from_host(float *v);
    void set_from_host(std::vector<float> &v);

    void set_to_host(float *v);
    void set_to_host(std::vector<float> &v);

    void set_const(float value);
    void clear();

    void set_random(float range);
    void set_random_xavier();

  public:
    void add(Tensor &rhs);
    void sub(Tensor &rhs);
    void mul(float value);

  public:
    void regularization_l1(float lambda);
    void regularization_l2(float lambda);

  public:
    void  set(unsigned int x, unsigned y, unsigned z, float value);
    float get(unsigned int x, unsigned y, unsigned z);

  public:
    bool is_valid();
    sGeometry get_geometry();
    void print();

  public:
    void save_to_file(std::string file_name);
    void load_from_file(std::string file_name);

  public:
    float *v;

  private:

    unsigned int m_w, m_h, m_d, m_size;

  public:
    void copy(Tensor &other);
    void copy(const Tensor &other);

    unsigned int to_idx(unsigned int x, unsigned y, unsigned z);

  public:
    void init(sGeometry geometry);
    void init(unsigned int w_, unsigned h_ = 1, unsigned d_ = 1);

  public:
    float rms(Tensor &rhs);

};

#endif
