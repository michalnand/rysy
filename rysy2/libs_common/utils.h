#ifndef _UTILS_H_
#define _UTILS_H_

#include <vector>

struct sPoint2D
{
    unsigned int x, y;
};

sPoint2D make_rectangle(unsigned int count);

void normalise(std::vector<float> &v, float min = 0.0, float max = 1.0);

#endif
