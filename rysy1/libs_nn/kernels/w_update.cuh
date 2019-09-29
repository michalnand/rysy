#ifndef _W_UPDATE_H_
#define _W_UPDATE_H_

#include "../tensor.h"

void w_update(Tensor &w, Tensor &w_grad, Tensor &m, Tensor &v,
              sHyperparameters hyperparameters);

#endif
