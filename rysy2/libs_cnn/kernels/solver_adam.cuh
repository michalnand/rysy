#ifndef _SOLVER_ADAM_H_
#define _SOLVER_ADAM_H_

#include <tensor.h>

void solver_adam(   Tensor &w, Tensor &w_grad, Tensor &m, Tensor &v,
                    float learning_rate, float lambda1, float lambda2);

#endif
