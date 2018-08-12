#ifndef _NN_STRUCT_H_
#define _NN_STRUCT_H_

struct sGeometry
{
  unsigned int w, h, d;
};


struct sHyperparameters
{
  float init_weight_range;
  float learning_rate;
  float lambda;
  float dropout;
  float noise;

  float beta1;
  float beta2;
  float epsilon;

  unsigned int minibatch_size;
};


#endif
