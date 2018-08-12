#include "rl_agent.h"


RLAgent::RLAgent()
{
  env                     = nullptr;
  actions_count           = 0;

  hyperparameters.alpha   = 0.0;
  hyperparameters.gamma   = 0.0;
  hyperparameters.epsilon = 0.0;
}

RLAgent::RLAgent(RLAgent& other)
{
  copy(other);
}

RLAgent::RLAgent(const RLAgent& other)
{
  copy(other);
}

RLAgent::RLAgent(RLEnvironment &env, sRLAgentHyperparameters hyperparameters)
{
  init(env, hyperparameters);
}

RLAgent::RLAgent(const RLEnvironment &env, sRLAgentHyperparameters hyperparameters)
{
  init(env, hyperparameters);
}

RLAgent::~RLAgent()
{

}

RLAgent& RLAgent::operator= (RLAgent& other)
{
  copy(other);
  return *this;
}

RLAgent& RLAgent::operator= (const RLAgent& other)
{
  copy(other);
  return *this;
}


void RLAgent::init(RLEnvironment &env, sRLAgentHyperparameters hyperparameters)
{
  this->env             = &env;
  this->hyperparameters = hyperparameters;
  this->actions_count   = this->env->get_actions_count();
}

void RLAgent::init(const RLEnvironment &env, sRLAgentHyperparameters hyperparameters)
{
  this->env             = (RLEnvironment*)&env;
  this->hyperparameters = hyperparameters;
  this->actions_count   = this->env->get_actions_count();
}

void RLAgent::copy(RLAgent &other)
{
  env               = other.env;
  actions_count     = other.actions_count;
  hyperparameters   = other.hyperparameters;
}

void RLAgent::copy(const RLAgent &other)
{
  env               = other.env;
  actions_count     = other.actions_count;
  hyperparameters   = other.hyperparameters;
}

void RLAgent::main()
{
  unsigned int action = rand()%actions_count;

  env->execute_action(action);
}
