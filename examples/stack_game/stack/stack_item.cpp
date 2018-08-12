#include "stack_item.h"

#include <stdio.h>
#include <stdlib.h>

StackItem::StackItem( unsigned int area_width, unsigned int area_height,
                      unsigned int width, unsigned int height)
{
  this->area_width = area_width;
  this->area_height = area_height;
  this->width = width;
  this->height = height;

  position = 0;

  move_type = 0;
  state = 5;

  x = (area_width - width)/2;
  y = (area_height - height)/2;
  target_x = x;
  target_y = y;

  values.resize(area_height);
  for (unsigned int j = 0; j < values.size(); j++)
  {
    values[j].resize(area_width);

    for (unsigned int i = 0; i < values[j].size(); i++)
      values[j][i] = 0.0;
  }

  other = nullptr;



  fill_values();
}

StackItem::StackItem(StackItem *other)
{
  this->other = other;


  this->area_width  = other->area_width;
  this->area_height = other->area_height;
  this->width  = other->width;
  this->height = other->height;


  move_type = other->move_type + 1;
  state = 2;

  position = 0;


  x = other->x;
  y = other->y;
  target_x = x;
  target_y = y;

  values.resize(area_height);
  for (unsigned int j = 0; j < values.size(); j++)
  {
    values[j].resize(area_width);

    for (unsigned int i = 0; i < values[j].size(); i++)
      values[j][i] = 0.0;
  }


  if ((move_type%2) ==0)
  {
    state = 0;
    x = rand()%(area_width - width);
  }
  else
  {
    state = 2;
    y = rand()%(area_width - width);
  }

  fill_values();
}


float StackItem::compute_overlap()
{
  float max_count = 0.0;
  float count = 0.0;
  for (unsigned int j = 0; j < values.size(); j++)
    for (unsigned int i = 0; i < values[j].size(); i++)
    {
      if (other->values[j][i] > 0.0)
        max_count+= 1.0;

      if (other->values[j][i] > 0.0)
      if (values[j][i] > 0.0)
        count+= 1.0;
    }

  return count/(max_count + 0.000000001);
}

void StackItem::process(unsigned int action)
{
  if (action == 1)
  {
    state = 4;
  }


  switch (state)
  {
    case 0:
            if (x < (area_width - width))
              x++;
            else
              state = 1;
            break;
    case 1:
            if (x > 0)
              x--;
            else
              state = 0;
            break;

    case 2:
            if (y < (area_height - height))
              y++;
            else
              state = 3;
            break;

    case 3:
            if (y > 0)
              y--;
            else
              state = 2;
            break;

    default:
            break;
  }

  fill_values();
}

void StackItem::print()
{
  for (unsigned int j = 0; j < values.size(); j++)
  {
    for (unsigned int i = 0; i < values[j].size(); i++)
    {
      if (values[j][i] > 0.0)
        printf("*");
      else
        printf(".");
      printf(" ");
    }
    printf("\n");
  }
  printf("\n");
}

void StackItem::fill_values()
{
  for (int j = 0; j < (int)values.size(); j++)
    for (int i = 0; i < (int)values[j].size(); i++)
      values[j][i] = 0.0;


  for (int j = 0; j < height; j++)
    for (int i = 0; i < width; i++)
    {
      int y_ = j + y;
      int x_ = i + x;


      if ( y_ < area_height && x_ < area_width )
      {
        values[y_][x_] = 0.5;
        if ((j == 0)||(i == 0)||(j+1 == height)||(i+1 == width))
          values[y_][x_] = 1.0;
      }
    }
}

/*
StackItem::StackItem()
{

}

StackItem::StackItem(   unsigned int area_width, unsigned int area_height,
                        unsigned int width, unsigned int height)
{
  other = nullptr;

  this->area_width = area_width;
  this->area_height = area_height;
  this->width = width;
  this->height = height;

  this->x = (area_width - width)/2;
  this->y = (area_height - height)/2;

  target_x = this->x;
  target_y = this->y;


  this->move_type = 0;

  state = 2;
  position = 0;


  values.resize(area_height);
  for (unsigned int j = 0; j < values.size(); j++)
  {
    values[j].resize(area_width);
    for (unsigned int i = 0; i < values[j].size(); i++)
      values[j][i] = 0.0;
  }

  set_position();
}


StackItem::StackItem(StackItem &other)
{
  this->area_width = other.area_width;
  this->area_height = other.area_height;
  this->width = other.width;
  this->height = other.height;

  this->move_type = other.move_type+1;

  target_x = other.x;
  target_y = other.y;

  state = 0;
  position = 0;

  if ((move_type%2) == 0)
  {
    this->x = (other.area_width - other.width)/2;
    this->y = 0;
    dimension_max = height;
    position_max  = area_height;
    position = math.rand()%(other.area_height - other.height);
  }
  else
  {
    this->x = 0;
    this->y = (other.area_height - other.height)/2;
    dimension_max = width;
    position_max  = area_width;
    position = math.rand()%(other.area_width - other.width);
  }

  values.resize(area_height);
  for (unsigned int j = 0; j < values.size(); j++)
  {
    values[j].resize(area_width);
    for (unsigned int i = 0; i < values[j].size(); i++)
      values[j][i] = 0.0;
  }

  set_position();
}

StackItem::~StackItem()
{

}

void StackItem::set_position()
{
  for (unsigned int j = 0; j < values.size(); j++)
    for (unsigned int i = 0; i < values[j].size(); i++)
      values[j][i] = 0.0;

  for (unsigned int j = 0; j < height; j++)
    for (unsigned int i = 0; i < width; i++)
      if ( (j + y) < values.size() && (i + x) < values[j].size() )
      {
        values[j + y][i + x] = 0.5;
        if ((j == 0)||(i == 0)||(j+1 == height)||(i+1 == width))
          values[j + y][i + x] = 1.0;
      }
}

void StackItem::print()
{
  for (unsigned int j = 0; j < values.size(); j++)
  {
    for (unsigned int i = 0; i < values[j].size(); i++)
    {
      if (values[j][i] > 0.0)
        printf("*");
      else
        printf(".");
      printf(" ");
    }
    printf("\n");
  }
  printf("\n");
}

void StackItem::process(unsigned int action)
{
  if (action == 1)
  {

    state = 2;
  }

  switch (state)
  {
    case 0:
            position++;
            if (position >= (position_max - dimension_max))
              state = 1;
            break;

    case 1:
            position--;
            if (position <= 0)
              state = 0;
            break;

    case 2:
            break;
  }

  if (state != 2)
  {
    if ((move_type%2) == 0)
    {
      this->x = target_x;
      this->y = position;
    }
    else
    {
      this->x = position;
      this->y = target_y;
    }
  }

  set_position();
}


float StackItem::compute_overlap(StackItem &other)
{
  float max_count = width*height;
  float overlap_count = 0.0;

  for (unsigned int j = 0; j < values.size(); j++)
    for (unsigned int i = 0; i < values[j].size(); i++)
      if (values[j][i] > 0.0)
      if (other.values[j][i] > 0.0)
        overlap_count++;

  return overlap_count/(max_count + 0.00000001);
}

int StackItem::min(int a, int b)
{
  if (a < b)
    return a;
  return b;
}

int StackItem::max(int a, int b)
{
  if (a > b)
    return a;
  return b;
}

void StackItem::process_overlap(StackItem *other)
{
  this->other = other;

  int _x     = min(x, other.x);
  int _x_end = min(x + width, other.x + other.width);

  int _y     = min(y, other.y);
  int _y_end = min(y + height, other.y + other.height);


  this->x = _x;
  this->y = _y;

  this->width  = max(_x_end - _x, 0);
  this->height = max(_y_end - _y, 0);


  if ((move_type%2) == 0)
  {
    this->x = target_x;
    this->y = y;
    position = y;
  }
  else
  {
    this->x = x;
    this->y = target_y;
    position = x;
  }


  for (unsigned int j = 0; j < values.size(); j++)
    for (unsigned int i = 0; i < values[j].size(); i++)
    {
      if (other.values[j][i] < 0.0001)
        values[j][i] = 0.0;
    }

  int width = 0;
  for (unsigned int j = 0; j < values.size(); j++)
  {
    unsigned int cnt = 0;
    for (unsigned int i = 0; i < values[0].size(); i++)
      if (values[j][i] > 0.0)
        cnt++;
    if (cnt > width)
      width = cnt;
  }

  int height = 0;
  for (unsigned int i = 0; i < values[0].size(); i++)
  {
    unsigned int cnt = 0;
    for (unsigned int j = 0; j < values.size(); j++)
      if (values[j][i] > 0.0)
        cnt++;
    if (cnt > height)
      height = cnt;
  }

  width = width - (other.x - x);
  height = height - (other.y - y);

  if (width < 0)
    width = 0;
  if (height < 0)
    height = 0;

  this->width = width;
  this->height = height;

}
*/
