#include <tensor_to_activity_map.h>

#include <json_config.h>
#include <iostream>

TensorToActivityMap::TensorToActivityMap()
{

}

TensorToActivityMap::TensorToActivityMap(Shape m_shape)
{
    init(m_shape);
}

TensorToActivityMap::~TensorToActivityMap()
{

}

TensorToActivityMap::TensorToActivityMap(TensorToActivityMap &other)
{
    copy(other);
}

TensorToActivityMap::TensorToActivityMap(const TensorToActivityMap &other)
{
    copy(other);
}

TensorToActivityMap& TensorToActivityMap::operator=(TensorToActivityMap &rhs)
{
    copy(rhs);
    return *this;
}

TensorToActivityMap& TensorToActivityMap::operator=(const TensorToActivityMap &rhs)
{
    copy(rhs);
    return *this;
}


void TensorToActivityMap::copy(TensorToActivityMap &other)
{
    this->m_shape     = other.m_shape;
    this->t_sum     = other.t_sum;

    this->v_sum     = other.v_sum;
    this->result    = other.result;
}

void TensorToActivityMap::copy(const TensorToActivityMap &other)
{
    this->m_shape     = other.m_shape;
    this->t_sum     = other.t_sum;

    this->v_sum     = other.v_sum;
    this->result    = other.result;
}

void TensorToActivityMap::init(Shape m_shape)
{
    this->m_shape = m_shape;

    t_sum.init(m_shape);

    v_sum.resize(this->m_shape.size());

    result.resize(this->m_shape.d());

    for (unsigned int ch = 0; ch < this->m_shape.d(); ch++)
    {
        result[ch].resize(this->m_shape.h());

        for (unsigned int y = 0; y < this->m_shape.h(); y++)
        {
            result[ch][y].resize(this->m_shape.w());
            for (unsigned int x = 0; x < this->m_shape.w(); x++)
                result[ch][y][x] = 0.0;
        }
    }

    clear();
}

void TensorToActivityMap::clear()
{
    t_sum.clear();

    for (unsigned int i = 0; i < v_sum.size(); i++)
        v_sum[i] = 0.0;

    for (unsigned int ch = 0; ch < this->m_shape.d(); ch++)
    for (unsigned int y = 0; y < this->m_shape.h(); y++)
    for (unsigned int x = 0; x < this->m_shape.w(); x++)
        result[ch][y][x] = 0.0;
}

void TensorToActivityMap::add(Tensor &v)
{
    t_sum.add(v);
}

void TensorToActivityMap::save(std::string output_path, unsigned int output_scale)
{

    compute_result();

    this->m_color_map = create_color_map(this->m_shape.d());

    unsigned int output_width    = m_shape.w()*output_scale;
    unsigned int output_height   = m_shape.h()*output_scale;

    {
        auto result_scaled = scale(output_scale, true);
        ImageSave image_save(output_width, output_height, false);
        image_save.save(output_path + "_max.png", result_scaled);
    }

    {
        auto result_scaled = scale(output_scale, false);
        ImageSave image_save(output_width, output_height, false);
        image_save.save(output_path + "_average.png", result_scaled);
    }


    JsonConfig output_json;

    output_json.result["shape"][0] = m_shape.w();
    output_json.result["shape"][1] = m_shape.h();
    output_json.result["shape"][2] = m_shape.d();


    for (unsigned int ch = 0; ch < m_shape.d(); ch++)
    for (unsigned int y = 0; y < m_shape.h(); y++)
    for (unsigned int x = 0; x < m_shape.w(); x++)
    {
        output_json.result["result"][ch][y][x] = result[ch][y][x];
    }

    output_json.save(output_path + ".json");

}

Shape TensorToActivityMap::shape()
{
    return m_shape;
}

void TensorToActivityMap::compute_result()
{
    t_sum.set_to_host(v_sum);


    //normalise into <0, 1> interval

    float min = v_sum[0];
    float max = v_sum[0];
    for (unsigned int i = 0; i < v_sum.size(); i++)
    {
        if (v_sum[i] < min)
            min = v_sum[i];
        if (v_sum[i] > max)
            max = v_sum[i];
    }

    float k = 0.0;
    float q = 0.0;
    if (max > min)
    {
        k = (1.0 - 0.0)/(max - min);
        q = 1.0 - k*max;
    }

    for (unsigned int i = 0; i < v_sum.size(); i++)
        v_sum[i] = v_sum[i]*k + q;


    //fill input
    for (unsigned int ch = 0; ch < m_shape.d(); ch++)
        for (unsigned int y = 0; y < m_shape.h(); y++)
            for (unsigned int x = 0; x < m_shape.w(); x++)
            {
                unsigned int in_idx = (ch*m_shape.h() + y)*m_shape.w() + x;
                result[ch][y][x] = v_sum[in_idx];
            }
}

std::vector<float> TensorToActivityMap::scale(unsigned int output_scale, bool max_color)
{
    unsigned int scaled_h = output_scale*m_shape.h();
    unsigned int scaled_w = output_scale*m_shape.w();

    std::vector<float> result_scaled(3*scaled_h * scaled_w );


    for (unsigned int y = 0; y < m_shape.h(); y++)
        for (unsigned int x = 0; x < m_shape.w(); x++)
        {
            std::vector<float> color;

            if (max_color)
                color = get_max_color(x, y);
            else
                color = get_average_color(x, y);

            for (unsigned int ch = 0; ch < 3; ch++)
                for (unsigned int ky = 0; ky < output_scale; ky++)
                    for (unsigned int kx = 0; kx < output_scale; kx++)
                    {
                        unsigned int out_idx    = (ch*scaled_h + (y*output_scale + ky))*scaled_w + x*output_scale + kx;
                        result_scaled[out_idx] = color[ch];
                    }
        }

    return result_scaled;
}

std::vector<std::vector<float>> TensorToActivityMap::create_color_map(unsigned int count)
{
    std::vector<std::vector<float>> result;

    float pi = 3.141592654;

    for (unsigned int i = 0; i < count; i++)
    {
        float phase = i*2.0*pi/count;

        std::vector<float> color(3);

        color[0] = (cos(phase + pi*0.0/3.0) + 1.0)/2.0;
        color[1] = (cos(phase + pi*1.0/3.0) + 1.0)/2.0;
        color[2] = (cos(phase + pi*2.0/3.0) + 1.0)/2.0;

        result.push_back(color);
    }

    return result;
}

std::vector<float> TensorToActivityMap::get_max_color(unsigned int x, unsigned int y)
{
    unsigned int max_ch = 0;
    for (unsigned int ch = 0; ch < m_shape.d(); ch++)
        if (result[ch][y][x] > result[max_ch][y][x])
            max_ch = ch;

    return m_color_map[max_ch];
}

std::vector<float> TensorToActivityMap::get_average_color(unsigned int x, unsigned int y)
{
    std::vector<float> color_result(3);

    float w_sum = 0.0;
    for (unsigned int ch = 0; ch < m_shape.d(); ch++)
    {
        float w = result[ch][y][x];
        w_sum+= w;

        color_result[0]+= w*m_color_map[ch][0];
        color_result[1]+= w*m_color_map[ch][1];
        color_result[2]+= w*m_color_map[ch][2];
    }

    if (w_sum > 0.0)
    {
        color_result[0]/= w_sum;
        color_result[1]/= w_sum;
        color_result[2]/= w_sum;
    }
    else
    {
        color_result[0] = 0.0;
        color_result[1] = 0.0;
        color_result[2] = 0.0;
    }

    return color_result;
}
