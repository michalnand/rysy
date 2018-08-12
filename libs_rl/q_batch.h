#ifndef _Q_BATCH_
#define _Q_BATCH_


#include <vector>
#include "rl_state.h"

class QbatchItem
{
  public:
    RLState state;

    float         reward;
    unsigned int  action;
    float         q_value;
};

class QBatch
{
  protected:
    unsigned int current_ptr;
    std::vector<QbatchItem> q_values;

  public:
    QBatch(unsigned int batch_size = 32);
    ~QBatch();

    void add(RLState &state, unsigned int action, float reward);
    bool is_full();
    void clear();
    void compute(float gamma);

    QbatchItem get_random();

    unsigned int size();
};


#endif
