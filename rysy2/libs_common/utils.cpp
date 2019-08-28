#include <utils.h>
#include <math.h>

sPoint2D make_rectangle(unsigned int count)
{
    unsigned int width = sqrt(count);

    while (count%width != 0)
        width++;

    unsigned int height = count/width;

    sPoint2D result;
    result.x = width;
    result.y = height;
    return result;
}

void normalise(std::vector<float> &v, float min, float max)
{
    float max_v = v[0];
    float min_v = v[0];
    for (unsigned int i = 0; i < v.size(); i++)
    {
      if (v[i] > max_v)
        max_v = v[i];

      if (v[i] < min_v)
        min_v = v[i];
    }

    float k = 0.0;
    float q = 0.0;

    if (max_v > min_v)
    {
      k = (max - min)/(max_v - min_v);
      q = max - k*max_v;
    }

    for (unsigned int i = 0; i < v.size(); i++)
    {
      v[i] = k*v[i] + q;
    }
}

float saturate(float value, float min, float max)
{
    if (value < min)
        value = min;
    if (value > max)
        value = max;

    return value;
}
