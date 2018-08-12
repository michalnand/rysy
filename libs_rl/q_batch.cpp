#include "q_batch.h"
#include <stdlib.h>


QBatch::QBatch(unsigned int batch_size)
{
  current_ptr = 0;
  q_values.resize(batch_size);

  for (unsigned int i = 0; i < q_values.size(); i++)
  {
    q_values[i].reward  = 0.0;
    q_values[i].q_value = 0.0;
    q_values[i].action  = 0;
  }
}


QBatch::~QBatch()
{

}

void QBatch::add(RLState &state, unsigned int action, float reward)
{
  if (current_ptr < q_values.size())
  {
    q_values[current_ptr].state   = state;
    q_values[current_ptr].action  = action;
    q_values[current_ptr].reward  = reward;
    q_values[current_ptr].q_value = 0.0;

    current_ptr++;
  }
}

bool QBatch::is_full()
{
  return (current_ptr >= q_values.size());
}

void QBatch::clear()
{
  current_ptr = 0;
}

void QBatch::compute(float gamma)
{
  for (int i = (q_values.size()-2); i >= 0; i--)
    q_values[i].q_value = q_values[i].reward + gamma*q_values[i+1].q_value;
}

QbatchItem QBatch::get_random()
{
    return q_values[rand()%q_values.size()];
}

unsigned int QBatch::size()
{
  return q_values.size();
}
