#ifndef _ROWS_SIGN_NOISE_LAYER_H_
#define _ROWS_SIGN_NOISE_LAYER_H_


#include <preprocessing_layer.h>

class RowsSignNoiseLayer: public PreprocessingLayer
{
  protected:
    Tensor noise;

  public:
    RowsSignNoiseLayer();
    RowsSignNoiseLayer(Json::Value parameters);
    virtual ~RowsSignNoiseLayer();

  public:
    void process(Tensor &output, Tensor &input);
};


#endif
