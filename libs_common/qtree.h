#ifndef _Q_TREE_H_
#define _Q_TREE_H_

#include <iostream>

#define Q_TREE_CHILD_COUNT   ((unsigned int)4)

template <class t_data> class QTree
{
  public:
    QTree<t_data> *m_next[Q_TREE_CHILD_COUNT];

    float m_x, m_y;

    t_data m_data;

    QTree(t_data data, float x, float y)
    {
      this->m_data = data;
      this->m_x = x;
      this->m_y = y;

      for (unsigned int i = 0; i < Q_TREE_CHILD_COUNT; i++)
        m_next[i] = nullptr;
    }

    ~QTree()
    {
      for (unsigned int i = 0; i < Q_TREE_CHILD_COUNT; i++)
      {
        if (m_next[i] != nullptr)
        {
          delete m_next[i];
          m_next[i] = nullptr;
        }
      }
    }

    void insert(t_data data, float x, float y)
    {
      if (y > m_y)
      {
        if (x > m_x)
          put_to(0, data, x, y);
        else
          put_to(1, data, x, y);
      }
      else
      {
        if (x > m_x)
          put_to(2, data, x, y);
        else
          put_to(3, data, x, y);
      }
    }

    t_data* get(float x, float y)
    {
      t_data *result = nullptr;
      t_data *result_tmp = nullptr;

      if (y > m_y)
      {
        if (x > m_x)
          result_tmp = get_from(0, x, y);
        else
          result_tmp = get_from(1, x, y);
      }
      else
      {
        if (x > m_x)
          result_tmp = get_from(2, x, y);
        else
          result_tmp = get_from(3, x, y);
      }

      if (result_tmp != nullptr)
        result = result_tmp;
      else
        result = &m_data;

      return result;
    }


    void traverse()
    {
      //std::cout << m_data << "\n";
      std::cout << get_depth() << "\n";

      for (unsigned int i = 0; i < Q_TREE_CHILD_COUNT; i++)
      {
        std::cout << " ";
        if (m_next[i] != nullptr)
          m_next[i]->traverse();
      }
    }

    unsigned int get_depth()
    {
      unsigned int depth_max = 0;
      for (unsigned int i = 0; i < Q_TREE_CHILD_COUNT; i++)
        if (m_next[i] != nullptr)
        {
          unsigned int tmp = m_next[i]->get_depth();
          if (tmp > depth_max)
            depth_max = tmp;
        }

      return depth_max+1;
    }

    protected:
      //put data to child node (or create new child node)
      void put_to(unsigned int idx, t_data &data, float x, float y)
      {
        if (m_next[idx] != nullptr)
          m_next[idx]->insert(data, x, y);
        else
          m_next[idx] = new QTree<t_data>(data, x, y);
      }

      //get data from corresponding child node, or return nullptr if no exist
      t_data* get_from(unsigned int idx, float x, float y)
      {
        if (m_next[idx] != nullptr)
          return m_next[idx]->get(x, y);
        else
          return nullptr;
      }
};



#endif
