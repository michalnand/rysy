#ifndef _LSTM_GATE_CUH_
#define _LSTM_GATE_CUH_

#include <tensor.h>


void lstm_gate_forward(     Tensor &c,
                            Tensor &forget,
                            Tensor &input,
                            Tensor &input_candidate,

                            Tensor &output_candidate,

                            Tensor &c_next,
                            Tensor &h_next);

void lstm_gate_backward( Tensor &h_next,
                        Tensor &h,
                        Tensor &x,

                        Tensor &control,
                        Tensor &update,

                        Tensor &error,

                        Tensor &h_error_back,
                        Tensor &update_error_back,
                        Tensor &control_error_back);


#endif
