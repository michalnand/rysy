#include "batch.h"

#include <stdlib.h>

Batch::Batch()
{
  input_geometry.w = 0;
  input_geometry.h = 0;
  input_geometry.d = 0;

  output_geometry.w = 0;
  output_geometry.h = 0;
  output_geometry.d = 0;
}

Batch::Batch(sGeometry input_geometry, sGeometry output_geometry, unsigned int batch_size, float noise_level)
{
  init(input_geometry, output_geometry, batch_size, noise_level);
}

Batch::Batch( unsigned int input_w, unsigned int input_h, unsigned int input_d,
              unsigned int output_w, unsigned int output_h, unsigned int output_d,
              unsigned int batch_size, float noise_level)
{
  sGeometry input_geometry;
  sGeometry output_geometry;

  input_geometry.w = input_w;
  input_geometry.h = input_h;
  input_geometry.d = input_d;

  output_geometry.w = output_w;
  output_geometry.h = output_h;
  output_geometry.d = output_d;

  init(input_geometry, output_geometry, batch_size, noise_level);
}

Batch::Batch( unsigned int input_d, unsigned int output_d,
              unsigned int batch_size, float noise_level)
{
  sGeometry input_geometry;
  sGeometry output_geometry;

  input_geometry.w = 1;
  input_geometry.h = 1;
  input_geometry.d = input_d;

  output_geometry.w = 1;
  output_geometry.h = 1;
  output_geometry.d = output_d;

  init(input_geometry, output_geometry, batch_size, noise_level);
}

Batch::~Batch()
{
  for (unsigned int i = 0; i < input.size(); i++)
    delete input[i];

  for (unsigned int i = 0; i < output.size(); i++)
    delete output[i];
}

void Batch::init(sGeometry input_geometry, sGeometry output_geometry, unsigned int batch_size, float noise_level)
{
  for (unsigned int i = 0; i < input.size(); i++)
    delete input[i];

  for (unsigned int i = 0; i < output.size(); i++)
    delete output[i];

  input.resize(batch_size);
  output.resize(batch_size);

  for (unsigned int i = 0; i < input.size(); i++)
    input[i] = new Tensor(input_geometry);

  for (unsigned int i = 0; i < output.size(); i++)
    output[i] = new Tensor(output_geometry);

  error.init(output_geometry);

  add_idx = 0;
  get_idx = 0;
  this->noise_level = noise_level;

  this->input_geometry = input_geometry;
  this->output_geometry = output_geometry;
}


bool Batch::add(float *output_host, float *input_host)
{
  if (add_idx < size())
  {
    input[add_idx]->set_from_host(input_host);
    output[add_idx]->set_from_host(output_host);

    /*
    TODO : tensor noise
    if (noise_level > 0.0)
      input[add_idx]->add_noise(noise_level);
    */

    add_idx++;

    return true;
  }
  else
    return false;
}

bool Batch::add(std::vector<float> &output_host, std::vector<float> &input_host)
{
  return add(&output_host[0], &input_host[0]);
}

void Batch::set(unsigned int idx)
{
  get_idx = idx%size();
}

unsigned int Batch::size()
{
  return input.size();
}

void Batch::clear()
{
  add_idx = 0;
  get_idx = 0;
}

void Batch::set_random()
{
  get_idx = rand()%size();
}

Tensor& Batch::get_input()
{
  return *input[get_idx];
}

Tensor& Batch::get_output()
{
  return *output[get_idx];

}
