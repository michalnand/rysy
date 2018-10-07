#include "preprocessing_layer.h"
#include <iostream>

PreprocessingLayer::PreprocessingLayer()
{

}

PreprocessingLayer::PreprocessingLayer(Json::Value parameters)
{
  this->parameters = parameters;

  std::cout << "creating layer " << parameters["type"].asString() << "\n";
}

PreprocessingLayer::PreprocessingLayer(PreprocessingLayer& other)
{
  copy(other);
}

PreprocessingLayer::PreprocessingLayer(const PreprocessingLayer& other)
{
  copy(other);
}

PreprocessingLayer::~PreprocessingLayer()
{

}

PreprocessingLayer& PreprocessingLayer::operator= (PreprocessingLayer& other)
{
  copy(other);

  return *this;
}

PreprocessingLayer& PreprocessingLayer::operator= (const PreprocessingLayer& other)
{
  copy(other);

  return *this;
}

void PreprocessingLayer::copy(PreprocessingLayer& other)
{
  parameters = other.parameters;
}

void PreprocessingLayer::copy(const PreprocessingLayer& other)
{
  parameters = other.parameters;
}




void PreprocessingLayer::process(Tensor &output, Tensor &input)
{
  output.copy(input);
}

sGeometry PreprocessingLayer::get_output_geometry(sGeometry input_geometry)
{
  return input_geometry;
}
