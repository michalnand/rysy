#ifndef _STACK_ITEM_H_
#define _STACK_ITEM_H_

#include <vector>

class StackItem
{
  protected:

    int area_width, area_height;
    int width, height;

    unsigned int move_type, state;

    int x, y, position;
    int target_x, target_y;

    StackItem *other;

  public:
    std::vector<std::vector<float>> values;

  public:
    StackItem(  unsigned int area_width, unsigned int area_height,
                unsigned int width, unsigned int height);

    StackItem(StackItem *other);

    float compute_overlap();
    void process(unsigned int action);
    void print();

  protected:
    void fill_values();
};


/*
  public:
    std::vector<std::vector<float>> values;
    unsigned int x, y;
    unsigned int target_x, target_y;
    unsigned int width, height;

  protected:
    StackItem *other;

    unsigned int area_width, area_height;

    unsigned int move_type;
    unsigned int state;

    int position, position_max, dimension_max;


  public:
    StackItem();

    StackItem(  unsigned int area_width, unsigned int area_height,
                unsigned int width, unsigned int height);

    StackItem(StackItem *other);

    ~StackItem();


    void process(unsigned int action);
    void print();

    float compute_overlap();

  protected:

    void process_overlap();
    void set_position();

    int min(int a, int b);
    int max(int a, int b);

};
*/

#endif
