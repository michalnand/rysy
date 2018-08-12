#ifndef _RL_ENVIRONMENT_H_
#define _RL_ENVIRONMENT_H_

#include "rl_state.h"

class RLEnvironment
{
  protected:
    RLState m_state;

    unsigned int m_actions_count;
    float m_reward;
    float m_score;


  public:
    RLEnvironment();
    RLEnvironment(std::string path);

    RLEnvironment(RLEnvironment& other);

    RLEnvironment(const RLEnvironment& other);

    virtual ~RLEnvironment();
    RLEnvironment& operator= (RLEnvironment& other);
    RLEnvironment& operator= (const RLEnvironment& other);

  protected:
    void copy(RLEnvironment &other);
    void copy(const RLEnvironment &other);

  public:
    RLState& state()
    {
      return m_state;
    }

    unsigned int get_actions_count()
    {
      return m_actions_count;
    }

    float reward()
    {
      return m_reward;
    }

    float score()
    {
      return m_score;
    }

    void reset_score()
    {
      m_score = 0.0;
    }

  public:
    virtual void execute_action(unsigned int action_id);

  public:
    virtual void save(std::string path);
    virtual void load(std::string path);

  public:
    virtual void print();
};

#endif
