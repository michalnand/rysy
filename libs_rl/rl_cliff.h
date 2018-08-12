#ifndef _RL_CLIFF_H_
#define _RL_CLIFF_H_

#include "rl_environment.h"


class RLCliff: public RLEnvironment
{
  private:
    unsigned int width, height;
    int x, y;

    std::vector<std::vector<float>> rewards;

    std::vector<std::vector<unsigned int>> visited;

  public:
    RLCliff();

    RLCliff(RLCliff& other);

    RLCliff(const RLCliff& other);

    virtual ~RLCliff();
    RLCliff& operator= (RLCliff& other);
    RLCliff& operator= (const RLCliff& other);

  protected:
    void copy(RLCliff &other);
    void copy(const RLCliff &other);

  public:
    void execute_action(unsigned int action_id);

    void print();
    void print_visited();

};


#endif
