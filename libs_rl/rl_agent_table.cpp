#include "rl_agent_table.h"



RLAgentTable::RLAgentTable()
             :RLAgent()
{

}

RLAgentTable::RLAgentTable(RLEnvironment &env, sRLAgentHyperparameters hyperparameters)
             :RLAgent(env, hyperparameters)
{
  init(env, hyperparameters);
}

RLAgentTable::RLAgentTable(const RLEnvironment &env, sRLAgentHyperparameters hyperparameters)
             :RLAgent(env, hyperparameters)
{
  init(env, hyperparameters);
}

RLAgentTable::RLAgentTable(RLAgentTable& other)
             :RLAgent(other)
{
  copy_table(other);
}

RLAgentTable::RLAgentTable(const RLAgentTable& other)
             :RLAgent(other)
{
  copy_table(other);
}

RLAgentTable::~RLAgentTable()
{

}

RLAgentTable& RLAgentTable::operator= (RLAgentTable& other)
{
  copy(other);
  copy_table(other);
  return *this;
}

RLAgentTable& RLAgentTable::operator= (const RLAgentTable& other)
{
  copy(other);
  copy_table(other);
  return *this;
}


void RLAgentTable::init(RLEnvironment &env, sRLAgentHyperparameters hyperparameters)
{
  RLAgent::init(env, hyperparameters);

  unsigned int state_size = this->env->state().size();
  q.resize(state_size);

  for (unsigned int j = 0; j < q.size(); j++)
  {
    q[j].resize(actions_count);
    for (unsigned int i = 0; i < q[j].size(); i++)
      q[j][i] = 0.0;
  }
}

void RLAgentTable::init(const RLEnvironment &env, sRLAgentHyperparameters hyperparameters)
{
  RLAgent::init(env, hyperparameters);

  unsigned int state_size = this->env->state().size();
  q.resize(state_size);

  for (unsigned int j = 0; j < q.size(); j++)
  {
    q[j].resize(actions_count);
    for (unsigned int i = 0; i < q[j].size(); i++)
      q[j][i] = 0.0;
  }
}

void RLAgentTable::copy_table(RLAgentTable &other)
{
  copy(other);
  q = other.q;
}

void RLAgentTable::copy_table(const RLAgentTable &other)
{
  copy(other);
  q = other.q;
}

void RLAgentTable::main()
{
  RLState       state   = env->state();
  unsigned int  action  = select_action(q[state.argmax()]);
  env->execute_action(action);

  batch.add(state, action, env->reward());


  if (batch.is_full())
  {
    batch.compute(hyperparameters.gamma);

    for (unsigned int i = 0; i < batch.size(); i++)
    {

      auto item       = batch.get_random();
      auto state_idx  = item.state.argmax();
      auto action     = item.action;

      q[state_idx][action] = (1.0 - hyperparameters.alpha)*q[state_idx][action] +
                                    hyperparameters.alpha*item.q_value;
    }

    batch.clear();
  }
}


unsigned int RLAgentTable::select_action(std::vector<float> &q_values)
{
  unsigned int max_i = 0;
  for (unsigned int i = 0; i < q_values.size(); i++)
    if (q_values[i] > q_values[max_i])
      max_i = i;

  if ( ((rand()%10000)/10000.0) < hyperparameters.epsilon)
    max_i = rand()%actions_count;

  return max_i;
}


void RLAgentTable::print_q()
{
  for (unsigned int j = 0; j < q.size(); j++)
  {
    for (unsigned int i = 0; i < q[j].size(); i++)
    {
      printf("%6.3f ", q[j][i]);
    }

    printf("\n");
  }

  printf("\n");
}
