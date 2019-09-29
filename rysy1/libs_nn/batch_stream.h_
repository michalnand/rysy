#ifndef _BATCH_STREAM_H_
#define _BATCH_STREAM_H_

#include <tensor.h>
#include <dataset_interface.h>

#include <thread>
#include <mutex>

class BatchStream
{
  protected:
    DatasetInterface *dataset;

    std::vector<Tensor*> input;
    std::vector<Tensor*> output;

    unsigned int read_ptr, write_ptr;
    sGeometry m_input_geometry, m_output_geometry;

  public:
    BatchStream();
    BatchStream(DatasetInterface &dataset, unsigned int batch_size, bool testing_batch = false);

    virtual ~BatchStream();

    void init(DatasetInterface &dataset, unsigned int batch_size, bool testing_batch = false);

  private:
    std::thread *filling_thread;
    std::mutex  mutex, mutex_free_space;
    bool m_running, m_testing_batch;

  public:
    void next();

    Tensor& get_input();
    Tensor& get_output();

  public:
    unsigned int size()
    {
      return input.size();
    }

    sGeometry input_geometry()
    {
      return m_input_geometry;
    }

    sGeometry output_geometry()
    {
      return m_output_geometry;
    }

  private:
    void filling_thread_func();
    bool is_full();

};

#endif
