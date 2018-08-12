#ifndef _LAYER_MEMORY_H_
#define _LAYER_MEMORY_H_

#include "tensor.h"

class LayerMemory
{
  public:
    Tensor output;
    Tensor error;

  public:
    LayerMemory();

    LayerMemory(LayerMemory& other);

    LayerMemory(const LayerMemory& other);

    virtual ~LayerMemory();

    LayerMemory& operator= (LayerMemory& other);

    LayerMemory& operator= (const LayerMemory& other);

  public:
    LayerMemory(sGeometry output_geometry);
    void init(sGeometry output_geometry);

    void clear();
    bool is_ready();

  protected:
    void copy(LayerMemory& other);
    void copy(const LayerMemory& other);

};


#endif
