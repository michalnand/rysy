#include "batch_stream.h"
#include <unistd.h>


BatchStream::BatchStream()
{
  read_ptr  = 0;
  write_ptr = 0;
  dataset   = nullptr;

  m_input_geometry.w = 0;
  m_input_geometry.h = 0;
  m_input_geometry.d = 0;

  m_output_geometry.w = 0;
  m_output_geometry.h = 0;
  m_output_geometry.d = 0;

  filling_thread = nullptr;
  m_running = false;
  m_testing_batch = false;
}

BatchStream::BatchStream(DatasetInterface &dataset, unsigned int batch_size, bool testing_batch)
{
  read_ptr  = 0;
  write_ptr = 0;
  this->dataset   = nullptr;

  m_input_geometry.w = 0;
  m_input_geometry.h = 0;
  m_input_geometry.d = 0;

  m_output_geometry.w = 0;
  m_output_geometry.h = 0;
  m_output_geometry.d = 0;

  filling_thread = nullptr;
  m_running = false;
  m_testing_batch = false;

  init(dataset, batch_size, testing_batch);
}

BatchStream::~BatchStream()
{
  m_running = false;
  mutex_free_space.unlock();

  if (filling_thread != nullptr)
  {
    filling_thread->join();
    delete filling_thread;
    filling_thread = nullptr;
  }

  for (unsigned int i = 0; i < input.size(); i++)
    delete input[i];

  for (unsigned int i = 0; i < output.size(); i++)
    delete output[i];
}

void BatchStream::init(DatasetInterface &dataset, unsigned int batch_size, bool testing_batch)
{
  this->dataset = &dataset;
  this->m_testing_batch = testing_batch;

  read_ptr  = 0;
  write_ptr = 1;

  m_input_geometry.w = dataset.get_width();
  m_input_geometry.h = dataset.get_height();
  m_input_geometry.d = dataset.get_channels();

  m_output_geometry.w = 1;
  m_output_geometry.h = 1;
  m_output_geometry.d = dataset.get_output_size();


  for (unsigned int i = 0; i < batch_size; i++)
  {
    input.push_back(new Tensor(input_geometry()));
    output.push_back(new Tensor(output_geometry()));
  }

  mutex_free_space.lock();
  m_running = true;

  filling_thread = new std::thread(&BatchStream::filling_thread_func, this);

  for (unsigned int i = 0; i < batch_size; i++)
    next();
}

void BatchStream::next()
{
  while (is_full() != false)
    __asm("nop");

  mutex.lock();
  read_ptr = (read_ptr+1)%size();
  mutex.unlock();
}

Tensor& BatchStream::get_input()
{
  return *input[read_ptr];
}

Tensor& BatchStream::get_output()
{
  return *output[read_ptr];
}

void BatchStream::filling_thread_func()
{
  while (m_running)
  {
      mutex.lock();

      sDatasetItem item;

      if (m_testing_batch)
        item = dataset->get_random_testing();
      else
        item = dataset->get_random_training();

      input[write_ptr]->set_from_host(item.input);
      output[write_ptr]->set_from_host(item.output);

      write_ptr = (write_ptr+1)%size();

      mutex.unlock();
  }
}


bool BatchStream::is_full()
{
  bool result = false;

  mutex.lock();
  if (write_ptr == read_ptr)
    result = true;
  mutex.unlock();

  return result;
}
