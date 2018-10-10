#ifndef _ROWS_DC_OFFSET_REMOVE_LAYER_H_
#define _ROWS_DC_OFFSET_REMOVE_LAYER_H_


#include <preprocessing_layer.h>

class RowsDCOffsetRemoveLayer: public PreprocessingLayer
{
  public:
    RowsDCOffsetRemoveLayer();
    RowsDCOffsetRemoveLayer(Json::Value parameters);
    virtual ~RowsDCOffsetRemoveLayer();

  public:
    void process(Tensor &output, Tensor &input);
};


#endif
