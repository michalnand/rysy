#include "layer_memory.h"


LayerMemory::LayerMemory()
{

}

LayerMemory::LayerMemory(LayerMemory& other)
{
  copy(other);
}

LayerMemory::LayerMemory(const LayerMemory& other)
{
  copy(other);
}

LayerMemory::~LayerMemory()
{

}

LayerMemory& LayerMemory::operator= (LayerMemory& other)
{
  copy(other);
  return *this;
}

LayerMemory& LayerMemory::operator= (const LayerMemory& other)
{
  copy(other);
  return *this;
}

LayerMemory::LayerMemory(sGeometry output_geometry)
{
  init(output_geometry);
}

void LayerMemory::init(sGeometry output_geometry)
{
  output.init(output_geometry);
  error.init(output_geometry);
}


void LayerMemory::clear()
{
  output.clear();
  error.clear();
}

void LayerMemory::copy(LayerMemory& other)
{
  output        = other.output;
  error         = other.error;
}

void LayerMemory::copy(const LayerMemory& other)
{
  output        = other.output;
  error         = other.error;
}
