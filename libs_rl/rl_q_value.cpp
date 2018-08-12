#include "rl_q_value.h"




RLQValue::RLQValue(float gamma)
         :RLState()
{
  m_action  = 0;
  m_gamma   = gamma;
  m_reward  = 0.0;
  m_q_value = 0.0;
}

RLQValue::RLQValue(RLQValue& other)
         :RLState(other)
{
  copy_q(other);
}

RLQValue::RLQValue(const RLQValue& other)
         :RLState(other)
{
  copy_q(other);
}

RLQValue::RLQValue(float gamma, unsigned int w, unsigned int h, unsigned int d)
         :RLState(w, h, d)
{
  m_action  = 0;
  m_gamma   = gamma;
  m_reward  = 0.0;
  m_q_value = 0.0;
}

RLQValue::~RLQValue()
{

}

RLQValue& RLQValue::operator= (RLQValue& other)
{
  copy_q(other);
  return *this;
}

RLQValue& RLQValue::operator= (const RLQValue& other)
{
  copy_q(other);
  return *this;
}


void RLQValue::copy_q(RLQValue &other)
{
  RLState::copy(other);

  m_action  = other.m_action;
  m_gamma   = other.m_gamma;
  m_reward  = other.m_reward;
  m_q_value = other.m_q_value;

}

void RLQValue::copy_q(const RLQValue &other)
{
  RLState::copy(other);

  m_action  = other.m_action;
  m_gamma   = other.m_gamma;
  m_reward  = other.m_reward;
  m_q_value = other.m_q_value;
}


void RLQValue::set_action(unsigned int action)
{
  m_action = action;
}

void RLQValue::set_reward(float reward)
{
  m_reward = reward;
}

void RLQValue::compute(RLQValue &next_value)
{
  m_q_value = next_value.m_q_value*m_gamma + m_reward;
}

void RLQValue::compute(const RLQValue &next_value)
{
  m_q_value = next_value.m_q_value*m_gamma + m_reward;
}

std::string RLQValue::as_string(unsigned int precision)
{
  std::string result;
  result = RLState::as_string(precision);
  result+= std::to_string(m_reward) + " ";
  result+= std::to_string(m_q_value) + " ";
  result+= std::to_string(m_action);

  result+= "\n\n";

  return result;
}
