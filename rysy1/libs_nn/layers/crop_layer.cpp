#include "crop_layer.h"

#include "kernels/crop_layer.cuh"


CropLayer::CropLayer()
        :Layer()
{

}

CropLayer::CropLayer(CropLayer& other)
        :Layer(other)
{
  copy_crop(other);
}

CropLayer::CropLayer(const CropLayer& other)
        :Layer(other)
{
  copy_crop(other);
}

CropLayer::~CropLayer()
{

}

CropLayer& CropLayer::operator= (CropLayer& other)
{
  copy(other);
  copy_crop(other);
  return *this;
}

CropLayer& CropLayer::operator= (const CropLayer& other)
{
  copy(other);
  copy_crop(other);
  return *this;
}

CropLayer::CropLayer(sGeometry input_geometry, sGeometry kernel_geometry, sHyperparameters hyperparameters)
        :Layer(input_geometry, kernel_geometry, hyperparameters)
{
  this->input_geometry.w = input_geometry.w;
  this->input_geometry.h = input_geometry.h;
  this->input_geometry.d = input_geometry.d;
 
  this->output_geometry.w = this->input_geometry.w - 2*kernel_geometry.w;
  this->output_geometry.h = this->input_geometry.h - 2*kernel_geometry.h;
  this->output_geometry.d = this->input_geometry.d;

  unsigned int output_size  = output_geometry.w*output_geometry.h*output_geometry.d;
  flops = output_size;

  layer_name = "CROP";
}

void CropLayer::copy_crop(CropLayer &other)
{
  (void)other;
}

void CropLayer::copy_crop(const CropLayer &other)
{
  (void)other;
}


void CropLayer::forward(Tensor &output, Tensor &input)
{
  crop_layer_forward(output, input);
}


void CropLayer::backward(LayerMemory &layer_mem_prev, LayerMemory &layer_mem, bool update_weights)
{
  crop_layer_backward(layer_mem_prev.error, layer_mem.error);
}
