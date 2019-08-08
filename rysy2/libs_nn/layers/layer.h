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

        Layer(Shape input_shape, Json::Value parameters);

        virtual ~Layer();

        virtual Layer& operator= (Layer& other);
        virtual Layer& operator= (const Layer& other);

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
        virtual bool is_activation() { return false;};
        
        virtual std::string asString();

        unsigned long int get_flops();
        unsigned long int get_trainable_parameters();

    protected:
        void init(Shape input_shape, Json::Value parameters);

    protected:
        Shape m_input_shape, m_output_shape;
        Json::Value m_parameters;
        bool m_training_mode;

        unsigned long int m_flops, m_trainable_parameters;
};

#endif
