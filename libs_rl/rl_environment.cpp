#include "rl_environment.h"



RLEnvironment::RLEnvironment()
{
  m_actions_count = 0;
  m_reward        = 0;
  m_score         = 0;
}

RLEnvironment::RLEnvironment(RLEnvironment& other)
{
  copy(other);
}

RLEnvironment::RLEnvironment(std::string path)
{
  load(path);
}

RLEnvironment::RLEnvironment(const RLEnvironment& other)
{
  copy(other);
}

RLEnvironment::~RLEnvironment()
{

}

RLEnvironment& RLEnvironment::operator= (RLEnvironment& other)
{
  copy(other);
  return *this;
}

RLEnvironment& RLEnvironment::operator= (const RLEnvironment& other)
{
  copy(other);
  return *this;
}

void RLEnvironment::copy(RLEnvironment &other)
{
  m_state         = other.m_state;
  m_actions_count = other.m_actions_count;
  m_reward        = other.m_reward;
  m_score         = other.m_score;
}

void RLEnvironment::copy(const RLEnvironment &other)
{
  m_state         = other.m_state;
  m_actions_count = other.m_actions_count;
  m_reward        = other.m_reward;
  m_score         = other.m_score;
}

void RLEnvironment::execute_action(unsigned int action_id)
{
  (void)action_id;
}

void RLEnvironment::load(std::string path)
{
  (void)path;
}

void RLEnvironment::save(std::string path)
{
  (void)path;
}

void RLEnvironment::print()
{

}
