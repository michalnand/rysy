#include "stack_game.h"

#include <math.h>
#include <iomanip>
#include <iostream>
#include <sstream>

StackGame::StackGame( unsigned int width,
                      unsigned int height,
                      bool visualisation_enabled)
          :IEnvironment()
{
  this->width  = width;
  this->height = height;

  visualisation = nullptr;
  if (visualisation_enabled)
    visualisation = new Visualisation;

  reset();
}

void StackGame::reset()
{
  for (unsigned int i = 0; i < items.size(); i++)
    delete items[i];
  items.clear();

  this->item_width  = width*0.3 + 1;
  this->item_height = height*0.3 + 1;

  items.push_back(new StackItem(width, height, item_width, item_height));
  items.push_back(new StackItem(items[0]));

  points = 0.0;

  unsigned int state_size = width*height*2;
  state.resize(state_size);

  actions_count = 2;

  geometry.w = width;
  geometry.h = height;
  geometry.d = 2;

  reward = 0.0;
  score  = 0.0;
  update_state();
}

StackGame::~StackGame()
{
  for (unsigned int i = 0; i < items.size(); i++)
    delete items[i];

  if (visualisation != nullptr)
    delete visualisation;
}

void StackGame::do_action(unsigned int action_id, IAgent *agent_id)
{
  (void)agent_id;

  unsigned int last = items.size()-1;

  float overlap = get_overlap();

  reward = 0.0;
  if (action_id == 1)
  {
    items[last]->process(action_id);

    points+= overlap;

    if (overlap > 0.95)
      reward+= 1.0;

    if (items.size() > 500)
      reset();
    else
    {
      items.push_back(new StackItem(items[last]));
      score = (100.0*points)/(items.size()-2.0);
    }
  }

  reward+= overlap;

  for (unsigned int i = 0; i < items.size(); i++)
    items[i]->process(0);

  update_state();

  if (visualisation != nullptr)
    draw();
}

void StackGame::print()
{
  unsigned int ptr = 0;
  for (unsigned int k = 0; k < geometry.d; k++)
  {
      for (unsigned int j = 0; j < geometry.h; j++)
      {
        for (unsigned int i = 0; i < geometry.w; i++)
        {
          printf("%4.1f ", state[ptr]);
          ptr++;
        }

        printf("\n");
      }

      printf("\n");
  }

  printf("reward = %6.3f\n", reward);
}

float StackGame::get_overlap()
{
  unsigned int last = items.size()-1;
  return items[last]->compute_overlap();
}

float angle = 0.0;
float iteration = 0.0;

void StackGame::draw()
{
  visualisation->start();

  // angle = angle + 1.0;
  iteration+= 1.0;

  unsigned int view_count = 50;
  for (unsigned int i = 0; i < items.size(); i++)
    if ((items.size() < view_count) || (i > (items.size()-view_count)))
      draw_item(i);

  std::stringstream stream;

  stream << std::fixed << std::setprecision(2) << score;

  std::string str_score;

  str_score+= "score : ";
  str_score+= stream.str();
  str_score+= "%";

  str_score+= "    height : " + std::to_string(items.size());

  visualisation->set_color(1.0, 1.0, 1.0);

  visualisation->print(-1.0, -1.0, -3.0, str_score);

  visualisation->finish();
/*
  unsigned int last = items.size()-1;
  for (int j = 0; j < height; j++)
  {
    for (int i = 0; i < width; i++)
    {
      if (items[last-1]->values[j][i] > 0.0)
        printf("B");
      else
      if (items[last]->values[j][i] > 0.0)
        printf("*");
      else
        printf(".");

      printf(" ");
    }
    printf("\n");
  }

  printf("\n");
*/
}

void StackGame::draw_item(unsigned int idx)
{
  float cube_size = 0.06;

  int height = items[idx]->values.size();
  int width  = items[idx]->values[0].size();

  visualisation->push();

  float x_ofs = 0.0;
  float y_ofs = -0.5*cube_size*(items.size() - idx) + 0.3;
  float z_ofs = -4.0;

  int steps = 100;

  visualisation->translate(x_ofs, y_ofs, z_ofs);

  visualisation->rotate(40.0, 45.0 + angle, 0.0);


  for (int j = 0; j < height; j++)
    for (int i = 0; i < width; i++)
    {
      if (items[idx]->values[j][i] > 0.0)
      {
        visualisation->push();
        float br = 0.5;

        if (items[idx]->values[j][i] > 0.999)
          br = 1.0;

        float x = (i - width/2.0)*1.2*cube_size;
        float y = 0.0;
        float z = -(j - height/2.0)*1.2*cube_size;



        float phase = (2.0*3.141592654*(idx%steps))/steps;

        float r = sin(phase + 0.0*2.0*3.141592654);
        float g = sin(phase + 0.3333*2.0*3.141592654);
        float b = sin(phase + 0.6666*2.0*3.141592654);

        visualisation->set_color(br*r, br*g, br*b);

        visualisation->translate(x, y, z);
        visualisation->draw_cube(cube_size);

        visualisation->pop();
      }
    }
  visualisation->pop();
}



void StackGame::update_state()
{
  unsigned int last = items.size()-1;
  unsigned int ptr = 0;


  for (unsigned int j = 0; j < height; j++)
    for (unsigned int i = 0; i < width; i++)
    {
      if (items[last]->values[j][i] > 0.0)
        state[ptr] = 1.0;
      else
        state[ptr] = -1.0;

      ptr++;
    }

  for (unsigned int j = 0; j < height; j++)
    for (unsigned int i = 0; i < width; i++)
    {
      if (items[last-1]->values[j][i] > 0.0)
        state[ptr] = 1.0;
      else
        state[ptr] = -1.0;

      ptr++;
    }
}
