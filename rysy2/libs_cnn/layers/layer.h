#ifndef _LAYER_H_
#define _LAYER_H_

#include <tensor.h>
#include <string>
#include <jsoncpp/json/value.h>

class Layer
{

    public:
        Layer();
        Layer(Layer& other);
        Layer(const Layer& other);

        Layer(Shape input_shape, Json::Value parameters, unsigned int max_time_steps = 1);

        virtual ~Layer();

        Layer& operator= (Layer& other);
        Layer& operator= (const Layer& other);

    protected:
        void copy(Layer& other);
        void copy(const Layer& other);

    public:
        Shape get_input_shape();
        Shape get_output_shape();

    public:

        virtual void reset();
        virtual void set_training_mode();
        virtual void unset_training_mode();

        virtual void forward(Tensor &output, Tensor &input);
        virtual void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights);

        virtual void save(std::string file_name_prefix);
        virtual void load(std::string file_name_prefix);

        virtual bool has_weights() { return false;};


    protected:
        void init(Shape input_shape, Json::Value parameters, unsigned int max_time_steps = 1);

    protected:
        Shape m_input_shape, m_output_shape;
        unsigned int m_max_time_steps;

        Json::Value m_parameters;

        bool m_training_mode;
};

#endif
