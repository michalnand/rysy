#ifndef _UTILS_H_
#define _UTILS_H_

#include <vector>

struct sPoint2D
{
    unsigned int x, y;
};

sPoint2D make_rectangle(unsigned int count);

void normalise(std::vector<float> &v, float min = 0.0, float max = 1.0);
void normalise(std::vector<std::vector<float>> &v, float min = 0.0, float max = 1.0);

float saturate(float value, float min, float max);

std::vector<std::vector<float>> upscale(std::vector<std::vector<float>> &source, unsigned int scaling_y, unsigned int scaling_x);

#endif
