#ifndef _RL_AGENT_H_
#define _RL_AGENT_H_

#include "rl_state.h"
#include "rl_environment.h"
#include "rl_agent_hyperparameters.h"

class RLAgent
{
  protected:
    RLEnvironment *env ;
    unsigned int actions_count;

    sRLAgentHyperparameters hyperparameters;

  public:
    RLAgent();

    RLAgent(RLEnvironment &env, sRLAgentHyperparameters hyperparameters);
    RLAgent(const RLEnvironment &env, sRLAgentHyperparameters hyperparameters);

    RLAgent(RLAgent& other);

    RLAgent(const RLAgent& other);

    virtual ~RLAgent();
    RLAgent& operator= (RLAgent& other);
    RLAgent& operator= (const RLAgent& other);


    void init(RLEnvironment &env, sRLAgentHyperparameters hyperparameters);
    void init(const RLEnvironment &env, sRLAgentHyperparameters hyperparameters);

  protected:
    void copy(RLAgent &other);
    void copy(const RLAgent &other);

  public:
    virtual void main();

};


#endif
