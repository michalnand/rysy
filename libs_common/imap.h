#ifndef _I_MAP_H_
#define _I_MAP_H_

#include "map_item.h"


class IMap
{
  public:
    IMap()
    {
      max_x = 0.0;
      max_y = 0.0;
    }

    virtual ~IMap()
    {

    }

    float get_max_x()
    {
      return max_x;
    }

    float get_max_y()
    {
      return max_y;
    }

    virtual void set(float x, float y, MapItem *item)
    {
      (void)x;
      (void)y;
      (void)item;
    }

    virtual MapItem* get(float x, float y)
    {
      (void)x;
      (void)y;
      return nullptr;
    }

    protected:
      float max_x, max_y;
};


#endif
