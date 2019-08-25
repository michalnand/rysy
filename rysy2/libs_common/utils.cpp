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
    float v_max = v[0];
    float v_min = v[0];
    for (unsigned int i = 0; i < v.size(); i++)
    {
        if (v[i] > v_max)
            v_max = v[i];

        if (v[i] < v_min)
            v_min = v[i];
    }

    float k = 0.0;
    float q = 0.0;

    if (v_max > v_min)
    {
        k = (max - min)/(v_max - v_min);
        q = max - k*v_max;
    } 

    for (unsigned int i = 0; i < v.size(); i++)
        v[i] = k*v[i] + q;
}
