#include "rows_dc_offset_remove_layer.h"
#include "kernels/rows_dc_offset_remove_layer.cuh"


RowsDCOffsetRemoveLayer::RowsDCOffsetRemoveLayer()
              :PreprocessingLayer()
{

}

RowsDCOffsetRemoveLayer::RowsDCOffsetRemoveLayer(Json::Value parameters)
              :PreprocessingLayer(parameters)
{

}

RowsDCOffsetRemoveLayer::~RowsDCOffsetRemoveLayer()
{

}


void RowsDCOffsetRemoveLayer::process(Tensor &output, Tensor &input, unsigned int augumentation)
{
  (void)augumentation;
  rows_dc_offset_remove_layer(output, input);
}
