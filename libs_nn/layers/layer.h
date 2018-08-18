#ifndef _LAYER_H_
#define _LAYER_H_

#include "../tensor.h"
#include "../layer_memory.h"
#include <string>


#define INIT_WEIGHT_RANGE_XAVIER_LIMIT    ((float) 0.00000001)


class Layer
{
  protected:

    sGeometry input_geometry, kernel_geometry, output_geometry;
    sHyperparameters hyperparameters;

    Tensor w, bias;
    Tensor m, v;
    Tensor w_grad;

    bool training_mode;
    std::string layer_name;

    unsigned long int flops;

  public:
    Layer();
    Layer(Layer& other);

    Layer(const Layer& other);
    virtual ~Layer();
    Layer& operator= (Layer& other);
    Layer& operator= (const Layer& other);

    Layer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters);

  protected:
    void copy(Layer &other);
    void copy(const Layer &other);

  public:
    void set_training_mode();
    void unset_training_mode();
    bool is_training_mode();

    virtual bool has_weights()
    {
      return false;
    }

    void save(std::string file_name_prefix);
    void load(std::string file_name_prefix);

    std::string get_info();
    void print();

    unsigned int get_required_memory();

    sGeometry get_input_geometry()
    {
      return input_geometry;
    }

    sGeometry get_kernel_geometry()
    {
      return kernel_geometry;
    }

    sGeometry get_output_geometry()
    {
      return output_geometry;
    }


    void set_learning_rate(float learning_rate);
    void set_lambda1(float lambda);
    void set_lambda2(float lambda);

    float get_flops()
    {
      return flops;
    }

  public:
    virtual void forward(Tensor &output, Tensor &input);
    virtual void backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights = false);



};

#endif
