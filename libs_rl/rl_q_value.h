#ifndef _RL_Q_VALUE_H_
#define _RL_Q_VALUE_H_

#include "rl_state.h"


class RLQValue: public RLState
{
  protected:
    unsigned int m_action;

    float m_gamma;
    float m_q_value, m_reward;

  public:

    RLQValue(float gamma = 0.9);
    RLQValue(RLQValue& other);
    RLQValue(const RLQValue& other);

    RLQValue(float gamma, unsigned int w, unsigned int h = 1, unsigned int d = 1);

    virtual ~RLQValue();
    RLQValue& operator= (RLQValue& other);
    RLQValue& operator= (const RLQValue& other);

    void set_action(unsigned int action);
    void set_reward(float reward);

    void compute(RLQValue &next_value);
    void compute(const RLQValue &next_value);

    std::string as_string(unsigned int precision = 3);


  public:
    float q()
    {
      return m_q_value;
    }

    float reward()
    {
      return m_reward;
    }

  protected:
    void copy_q(RLQValue &other);
    void copy_q(const RLQValue &other);

};

#endif
