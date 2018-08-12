#include <rl_cliff.h>
#include <rl_agent_table.h>

int main()
{
  srand(time(NULL));

  RLCliff env;

  sRLAgentHyperparameters hyperparameters;

  hyperparameters.alpha     = 0.01;
  hyperparameters.gamma     = 0.9;
  hyperparameters.epsilon   = 0.2;

  RLAgentTable agent(env, hyperparameters);

  for (unsigned int i = 0; i < 1000000; i++)
    agent.main();


  env.reset_score();
  env.print_visited();


  while (1)
  {
    env.print();
    agent.main();
  }


  return 0;
}
