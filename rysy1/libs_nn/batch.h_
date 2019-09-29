#ifndef _BATCH_H_
#define _BATCH_H_

#include <tensor.h>

class Batch
{
  protected:
    std::vector<Tensor*> input;
    std::vector<Tensor*> output;
    Tensor error;

    unsigned int add_idx, get_idx;
    float noise_level;

  public:
    sGeometry input_geometry, output_geometry;

  public:
    Batch();

    Batch(sGeometry input_geometry, sGeometry output_geometry,
          unsigned int batch_size, float noise_level = 0.0);

    Batch(  unsigned int input_w, unsigned int input_h, unsigned int input_d,
            unsigned int output_w, unsigned int output_h, unsigned int output_d,
            unsigned int batch_size, float noise_level = 0.0);

    Batch( unsigned int input_d, unsigned int output_d,
           unsigned int batch_size, float noise_level = 0.0);

    ~Batch();

  public:
    void init(sGeometry input_geometry, sGeometry output_geometry, unsigned int batch_size, float noise_level);

  public:

    bool add(float *output_host, float *input_host);
    bool add(std::vector<float> &output_host, std::vector<float> &input_host);

    void clear();

    void set_random();
    void set(unsigned int idx);
    unsigned int size();

    Tensor& get_input();
    Tensor& get_output();
};

#endif
