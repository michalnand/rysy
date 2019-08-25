#ifndef _MODEL_H_
#define _MODEL_H_

#include <shape.h>
#include <vector>
#include <string>

#include <tensor.h>

class Model
{
    public:
        Model();
        Model(const Model &other);
        Model& operator = (const Model &other);

        virtual ~Model();

    public:
        Shape input_shape();
        Shape output_shape();
        std::string name();

    public:
        unsigned int get_models_count();
        Model& get_model(unsigned int idx);

    public:
        std::string asString(bool full = false);
        void print();
        void print_full();

    public:
        virtual void forward(Tensor &output, Tensor &input);
        virtual void train(Tensor &target_output, Tensor &input);
        virtual void train_from_error(Tensor &error_back, Tensor &error);

    public:
        virtual Model& add(Model &model);
        void compile();


    protected:
        void copy_model(const Model &other);

    protected:
        Shape m_input_shape, m_output_shape;

        std::vector<Model> m_models;
        std::string m_name;
};

#endif
