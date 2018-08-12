#ifndef _RL_AGENT_TABLE_H_
#define _RL_AGENT_TABLE_H_

#include "rl_agent.h"
#include "q_batch.h"

class RLAgentTable : public RLAgent
{
  protected:

    std::vector<std::vector<float>> q;
    QBatch batch;


  public:
    RLAgentTable();

    RLAgentTable(RLEnvironment &env, sRLAgentHyperparameters hyperparameters);
    RLAgentTable(const RLEnvironment &env, sRLAgentHyperparameters hyperparameters);

    RLAgentTable(RLAgentTable& other);

    RLAgentTable(const RLAgentTable& other);

    virtual ~RLAgentTable();
    RLAgentTable& operator= (RLAgentTable& other);
    RLAgentTable& operator= (const RLAgentTable& other);


    void init(RLEnvironment &env, sRLAgentHyperparameters hyperparameters);
    void init(const RLEnvironment &env, sRLAgentHyperparameters hyperparameters);

  protected:
    void copy_table(RLAgentTable &other);
    void copy_table(const RLAgentTable &other);

  public:
    void main();
    void print_q();

  protected:
    unsigned int select_action(std::vector<float> &q_values);

};


#endif
