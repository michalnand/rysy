#include <layers/fir_layer.h>
#include <iostream>

FirLayer::FirLayer()
        :Layer()
{
    time_step_idx = 0;
}

FirLayer::FirLayer(FirLayer& other)
        :Layer(other)
{
    copy_fir_layer(other);
}

FirLayer::FirLayer(const FirLayer& other)
        :Layer(other)
{
    copy_fir_layer(other);
}

FirLayer::FirLayer(Shape input_shape, Json::Value parameters)
        :Layer(input_shape, parameters)
{
    init_fir_layer();
}

FirLayer::~FirLayer()
{

}

FirLayer& FirLayer::operator= (FirLayer& other)
{
    copy(other);
    copy_fir_layer(other);

    return *this;
}

FirLayer& FirLayer::operator= (const FirLayer& other)
{
    copy(other);
    copy_fir_layer(other);

    return *this;
}


void FirLayer::copy_fir_layer(FirLayer &other)
{
    this->h                     = other.h;
    this->error_h               = other.error_h;
    this->time_sequence_length  = other.time_sequence_length;
    this->time_step_idx         = other.time_step_idx;
}

void FirLayer::copy_fir_layer(const FirLayer &other)
{
    this->h                     = other.h;
    this->error_h               = other.error_h;
    this->time_sequence_length  = other.time_sequence_length;
    this->time_step_idx         = other.time_step_idx;
}


void FirLayer::reset()
{
    for (unsigned int i = 0; i < h.size(); i++)
        h[i].clear();

    for (unsigned int i = 0; i < error_h.size(); i++)
        error_h[i].clear();

    time_step_idx = 0;
}


void FirLayer::forward(Tensor &output, Tensor &input)
{
    #ifdef RYSY_DEBUG

    if (output.shape() != m_output_shape)
    {
        std::cout << "FirLayer::forward : inconsistent output shape ";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "FirLayer::forward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    if (time_step_idx < time_sequence_length)
    {
        h[time_step_idx] = input;

        output.concatenate_time_sequence(h);

        time_step_idx++;
    }
    #ifdef RYSY_DEBUG
    else
    {
        std::cout << "FirLayer::forward : exceed time_sequence_length " << time_step_idx << " expected " << time_sequence_length << "\n";
    }
    #endif
}

void FirLayer::backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights)
{
    (void)output;
    (void)input;
    (void)update_weights;

    #ifdef RYSY_DEBUG

    if (error_back.shape() != m_input_shape)
    {
        std::cout << "FirLayer::backward : inconsistent error_back shape\n";
        error_back.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (error.shape() != m_output_shape)
    {
        std::cout << "FirLayer::backward : inconsistent error shape\n";
        error.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    if (input.shape() != m_input_shape)
    {
        std::cout << "FirLayer::backward : inconsistent input shape\n";
        input.shape().print();
        std::cout << " : ";
        m_input_shape.print();
        std::cout << "\n";
        return;
    }

    if (output.shape() != m_output_shape)
    {
        std::cout << "FirLayer::backward : inconsistent output shape\n";
        output.shape().print();
        std::cout << " : ";
        m_output_shape.print();
        std::cout << "\n";
        return;
    }

    #endif

    error.split_time_sequence(error_h);

    time_step_idx--;

    error_back = error_h[time_step_idx];
}

std::string FirLayer::asString()
{
    std::string result;

    result+= "FIR LAYER\t";
    result+= "[" + std::to_string(m_input_shape.w()) + " " + std::to_string(m_input_shape.h()) + " " + std::to_string(m_input_shape.d()) + "]\t";
    result+= "[" + std::to_string(m_output_shape.w()) + " " + std::to_string(m_output_shape.h()) + " " + std::to_string(m_output_shape.d()) + " " + std::to_string(m_output_shape.t()) + "]\t";
    result+= "[" + std::to_string(get_trainable_parameters()) + " " + std::to_string(get_flops()) + "]\t";
    return result;
}

void FirLayer::init_fir_layer()
{
    time_step_idx        = 0;
    time_sequence_length = m_parameters["hyperparameters"]["time_sequence_length"].asInt();

    m_output_shape.set(m_input_shape.w(), m_input_shape.h(), m_input_shape.d(), time_sequence_length);

    this->m_flops                   = m_output_shape.size();

    h.resize(time_sequence_length);
    for (unsigned int i = 0; i < h.size(); i++)
        h[i].init(m_input_shape);

    error_h.resize(time_sequence_length);
    for (unsigned int i = 0; i < h.size(); i++)
        error_h[i].init(m_output_shape);
}
