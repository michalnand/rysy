#include "rl_cliff.h"
#include <iostream>


RLCliff::RLCliff()
        :RLEnvironment()
{

  width   = 16;
  height  = 5;
  x       = 0;
  y       = 0;

  m_state.init(width, height);
  m_state.set_element(1.0, x, y);

  m_actions_count = 4;

  rewards.resize(height);

  for (unsigned int j = 0; j < rewards.size(); j++)
  {
    rewards[j].resize(width);

    for (unsigned int i = 0; i < width; i++)
      rewards[j][i] = 0.0;
  }

  for (unsigned int i = 1; i < width-1; i++)
    rewards[0][i] = -1.0;

  rewards[0][width-1] = 1.0;


  visited.resize(height);
  for (unsigned int j = 0; j < visited.size(); j++)
  {
    visited[j].resize(width);
    for (unsigned int i = 0; i < width; i++)
      visited[j][i] = 0;
  }


}

RLCliff::RLCliff(RLCliff& other)
        :RLEnvironment(other)
{
  copy(other);
}

RLCliff::RLCliff(const RLCliff& other)
        :RLEnvironment(other)
{
  copy(other);
}

RLCliff::~RLCliff()
{

}

RLCliff& RLCliff::operator= (RLCliff& other)
{
  copy(other);
  return *this;
}

RLCliff& RLCliff::operator= (const RLCliff& other)
{
  copy(other);
  return *this;
}

void RLCliff::copy(RLCliff &other)
{
  RLEnvironment::copy(other);

  width   = other.width;
  height  = other.height;
  x       = other.x;
  y       = other.y;

  m_state.init(width, height);
  m_state.set_element(1.0, x, y);

  m_actions_count = other.m_actions_count;
  rewards         = other.rewards;
  visited         = other.visited;
}

void RLCliff::copy(const RLCliff &other)
{
  RLEnvironment::copy(other);

  RLEnvironment::copy(other);

  width   = other.width;
  height  = other.height;
  x       = other.x;
  y       = other.y;

  m_state.init(width, height);
  m_state.set_element(1.0, x, y);

  m_actions_count = other.m_actions_count;
  rewards         = other.rewards;
  visited         = other.visited;
}


void RLCliff::execute_action(unsigned int action_id)
{
  switch (action_id)
  {
    case 0: x = x + 1; break;
    case 1: x = x - 1; break;
    case 2: y = y + 1; break;
    case 3: y = y - 1; break;
  }


  if (x >= (int)(width-1))
    x = width - 1;
  if (x < 0)
    x = 0;

  if (y >= (int)(height-1))
    y = height - 1;
  if (y < 0)
    y = 0;

  m_reward = rewards[y][x];

  visited[y][x]++;


  if (m_reward < -0.00001)
  {
    y = 0;
    x = 0;
  }


  if (m_reward > 0.00001)
  {
    y = 0;
    x = 0;
  }

  m_score+= m_reward;

  m_state.clear();
  m_state.set_element(1.0, x, y);
}


void RLCliff::print()
{
  for (unsigned int j = 0; j < height; j++)
  {
    for (unsigned int i = 0; i < width; i++)
    {
      if (((int)j == y)&&((int)i == x))
        std::cout << "*";
      else if (rewards[j][i] < 0.0)
        std::cout << "!";
      else if (rewards[j][i] > 0.0)
        std::cout << "T";
      else
        std::cout << ".";
      std::cout << " ";
    }
    std::cout << "\n";
  }

  std::cout << "score = " << m_score << "\n";

  std::cout << "\n";
}

void RLCliff::print_visited()
{
  unsigned int sum = 0;
  for (unsigned int j = 0; j < height; j++)
    for (unsigned int i = 0; i < width; i++)
      sum+= visited[j][i];

  for (unsigned int j = 0; j < height; j++)
  {
    for (unsigned int i = 0; i < width; i++)
      printf("%6.3f ", visited[j][i]*100.0/sum);
    printf("\n");
  }

  printf("\n");
}
