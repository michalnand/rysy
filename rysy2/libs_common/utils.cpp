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


void normalise(std::vector<std::vector<float>> &v, float min, float max)
{
    float max_v = v[0][0];
    float min_v = v[0][0];

    for (unsigned int y = 0; y < v.size(); y++)
    for (unsigned int x = 0; x < v[y].size(); x++)
    {
      if (v[y][x] > max_v)
        max_v = v[y][x];

      if (v[y][x] < min_v)
        min_v = v[y][x];
    }
 
    float k = 0.0;
    float q = 0.0;

    if (max_v > min_v)
    {
      k = (max - min)/(max_v - min_v);
      q = max - k*max_v;
    }

    for (unsigned int y = 0; y < v.size(); y++)
    for (unsigned int x = 0; x < v[y].size(); x++)
    {
      v[y][x] = k*v[y][x] + q;
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


std::vector<std::vector<float>> upscale(std::vector<std::vector<float>> &source, unsigned int scaling_y, unsigned int scaling_x)
{
    unsigned int input_width   = source[0].size();
    unsigned int input_height  = source.size();

    unsigned int output_width  = scaling_x*source[0].size();
    unsigned int output_height = scaling_y*source.size();


    std::vector<std::vector<float>> result;
    result.resize(output_height);
    for (unsigned int y = 0; y < output_height; y++)
    {
        result[y].resize(output_width);
        for (unsigned int x = 0; x < output_width; x++)
            result[y][x] = 0.0;
    }

    for (unsigned int y = 0; y < input_height - 1; y++)
        for (unsigned int x = 0; x < input_width - 1; x++)
        {
            float x00 = source[y + 0][x + 0];
            float x01 = source[y + 0][x + 1];
            float x10 = source[y + 1][x + 0];
            float x11 = source[y + 1][x + 1];

            for (unsigned int ky = 0; ky < scaling_y; ky++)
                for (unsigned int kx = 0; kx < scaling_x; kx++)
                {
                    float x_ = kx*1.0/(scaling_x - 1);
                    float y_ = ky*1.0/(scaling_y - 1);

                    float v = 0.0;

                    v+= (1.0 - x_)* (1.0 - y_)*x00;
                    v+=        x_ * (1.0 - y_)*x01;
                    v+= (1.0 - x_)*        y_ *x10;
                    v+=        x_ *        y_ *x11;

                    result[y*scaling_y + ky][x*scaling_x + kx] = v;
                }

        }

    return result;
}
