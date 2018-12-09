#ifndef _GRU_GATE_LAYER_CUH_
#define _GRU_GATE_LAYER_CUH_

#include "../tensor.h"

void gru_gate_forward(Tensor &output, Tensor &state, Tensor &state_update, Tensor &gate);

void gru_gate_backward( Tensor &error,
                        Tensor &state,
                        Tensor &state_update,
                        Tensor &gate,
                        Tensor &error_back_state,
                        Tensor &error_back_state_update,
                        Tensor &error_back_gate);

#endif
