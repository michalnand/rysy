#ifndef _GRU_GATE_CUH_
#define _GRU_GATE_CUH_

#include <tensor.h>


void gru_gate_forward(  Tensor &h_next,
                        Tensor &h,
                        Tensor &x,

                        Tensor &control,
                        Tensor &update);

void gru_gate_backward( Tensor &h_next,
                        Tensor &h,
                        Tensor &x,

                        Tensor &control,
                        Tensor &update,

                        Tensor &error,

                        Tensor &h_error_back,
                        Tensor &update_error_back,
                        Tensor &control_error_back);


#endif
