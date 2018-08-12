#ifndef _ARRAY_DYNAMIC_H_
#define _ARRAY_DYNAMIC_H_

#include "array_interface.h"

template<class t_type> class ArrayDynamic: public IArray<t_type>
{
  private:
    unsigned int array_size;
    t_type *arr;

  public:
    ArrayDynamic()
    {
      array_size = 0;
      arr = nullptr;
    }

    ArrayDynamic(unsigned int size)
    {
      array_size = size;
      arr = new t_type[array_size];
    }

    ArrayDynamic(unsigned int size, t_type &init_value)
    {
      array_size = size;
      arr = new t_type[array_size];
      for (unsigned int i = 0; i < array_size; i++)
        arr[i] = init_value;
    }

    ArrayDynamic(const t_type &init_value)
    {
      array_size = size;
      arr = new t_type[array_size];
      for (unsigned int i = 0; i < array_size; i++)
        arr[i] = init_value;
    }

    ArrayDynamic(ArrayDynamic<t_type> &rhs)
    {
      array_size = rhs.size();
      arr = new t_type[array_size];
      for (unsigned int i = 0; i < array_size; i++)
        arr[i] = rhs[i];
    }

    virtual ~ArrayDynamic()
    {
      if (arr != nullptr)
      {
        delete arr;
        array_size = 0;
      }
    }


    int init(unsigned int size)
    {
      if (arr != nullptr)
        return -1;

      array_size = size;
      arr = new t_type[array_size];

      return array_size;
    }


    int init(unsigned int size, t_type &init_value)
    {
      if (arr != nullptr)
        return -1;

      array_size = size;
      arr = new t_type[array_size];

      for (unsigned int i = 0; i < array_size; i++)
        arr[i] = init_value;

      return array_size;
    }

    int init(unsigned int size, const t_type &init_value)
    {
      if (arr != nullptr)
        return -1;

      array_size = size;
      arr = new t_type[array_size];

      for (unsigned int i = 0; i < array_size; i++)
        arr[i] = init_value;

      return array_size;
    }

    void set(t_type &value)
    {
      for (unsigned int i = 0; i < array_size; i++)
        arr[i] = value;
    }

    void set(const t_type &value)
    {
      for (unsigned int i = 0; i < array_size; i++)
        arr[i] = value;
    }

    ArrayDynamic<t_type> operator =(ArrayDynamic<t_type> &rhs)
    {
      for (unsigned int i = 0; i < array_size; i++)
        arr[i] = rhs[i];

      return *this;
    }

    t_type& operator [](unsigned int rhs)
    {
      return arr[rhs];
    }

    bool operator !=(ArrayDynamic<t_type> &rhs)
    {
      for (unsigned int i = 0; i < array_size; i++)
        if (arr[i] != rhs[i])
          return true;

      return false;
    }

    bool operator ==(ArrayDynamic<t_type> &rhs)
    {
      for (unsigned int i = 0; i < array_size; i++)
        if (arr[i] != rhs[i])
          return false;

      return true;
    }

    unsigned int size()
    {
      return array_size;
    }
};


#endif
