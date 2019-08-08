#include <tensor_to_activity_map.h>

#include <log.h>

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
    result.resize(this->m_shape.w()*this->m_shape.h());

    clear();
}

void TensorToActivityMap::clear()
{
    t_sum.clear();

    for (unsigned int i = 0; i < v_sum.size(); i++)
        v_sum[i] = 0.0;

    for (unsigned int i = 0; i < result.size(); i++)
        result[i] = 0.0;
}

void TensorToActivityMap::add(Tensor &v)
{
    t_sum.add(v);
}

void TensorToActivityMap::save(std::string output_path, unsigned int output_scale)
{
    compute_result();

    unsigned int output_width    = m_shape.w()*output_scale;
    unsigned int output_height   = m_shape.h()*output_scale;

    auto result_scaled = scale(output_scale);

    ImageSave image_save(output_width, output_height, true);
    image_save.save(output_path + ".png", result_scaled);



    Log output_log(output_path + ".txt");
    for (unsigned int y = 0; y < m_shape.h(); y++)
    {
        for (unsigned int x = 0; x < m_shape.w(); x++)
        {
            unsigned int out_idx = y*m_shape.w() + x;
            output_log << result[out_idx] << " ";
        }
        output_log << "\n";
    }
    output_log << "\n";
}

Shape TensorToActivityMap::shape()
{
    return m_shape;
}

void TensorToActivityMap::compute_result()
{
    t_sum.set_to_host(v_sum);

    for (unsigned int i = 0; i < result.size(); i++)
        result[i] = 0.0;

    //fill input
        for (unsigned int y = 0; y < m_shape.h(); y++)
            for (unsigned int x = 0; x < m_shape.w(); x++)
            {
                float max = -100000.0;
                for (unsigned int ch = 0; ch < m_shape.d(); ch++)
                {
                    unsigned int in_idx     = (ch*m_shape.h() + y)*m_shape.w() + x;
                    if (v_sum[in_idx] > max)
                        max = v_sum[in_idx];
                }

                unsigned int out_idx    = y*m_shape.w() + x;
                result[out_idx] = max;
            }

    //normalise into <0, 1> interval

    float min = result[0];
    float max = result[0];
    for (unsigned int i = 0; i < result.size(); i++)
    {
        if (result[i] < min)
            min = result[i];
        if (result[i] > max)
            max = result[i];
    }

    float k = 0.0;
    float q = 0.0;
    if (max > min)
    {
        k = (1.0 - 0.0)/(max - min);
        q = 1.0 - k*max;
    }

    for (unsigned int i = 0; i < result.size(); i++)
        result[i] = result[i]*k + q;
}

std::vector<float> TensorToActivityMap::scale(unsigned int output_scale)
{
    unsigned int scaled_h = output_scale*m_shape.h();
    unsigned int scaled_w = output_scale*m_shape.w();
    std::vector<float> result_scaled( scaled_h * scaled_w );

    for (unsigned int y = 0; y < m_shape.h(); y++)
        for (unsigned int x = 0; x < m_shape.w(); x++)
        {
            unsigned int in_idx    = y*m_shape.w() + x;
            float v = result[in_idx];

            for (unsigned int ky = 0; ky < output_scale; ky++)
                for (unsigned int kx = 0; kx < output_scale; kx++)
                {
                    unsigned int out_idx    = (y*output_scale + ky)*scaled_h + x*output_scale + kx;

                    result_scaled[out_idx] = v;
                }
        }

    return result_scaled;
}
