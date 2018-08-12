#ifndef _STACK_GAME_H_
#define _STACK_GAME_H_

#include "stack_item.h"
#include <visualisation.h>
#include <ienvironment.h>

class StackGame: public IEnvironment
{
  private:
    std::vector<StackItem*> items;

    unsigned int width, height;
    unsigned int item_width, item_height;

    Visualisation *visualisation;

    float points;

  public:
    StackGame(unsigned int width = 21,
              unsigned int height = 21,
              bool visualisation_enabled = true);

    ~StackGame();

    void reset();
    void do_action(unsigned int action_id, IAgent *agent_id = nullptr);
    void print();

    float get_overlap();

  private:
    void draw();
    void draw_item(unsigned int idx);

    void update_state();
};

#endif
