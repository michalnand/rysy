#ifndef _CNN_H_
#define _CNN_H_

#include <log.h>

#include <tensor.h>
#include <layers/layer.h>



class CNN
{
    public:
        CNN();
        CNN(CNN& other);
        CNN(const CNN& other);

        CNN(std::string json_file_name, Shape input_shape = {0, 0, 0}, Shape output_shape = {0, 0, 0});
        CNN(Json::Value json_config, Shape input_shape = {0, 0, 0}, Shape output_shape = {0, 0, 0});
        CNN(Shape input_shape, Shape output_shape, float learning_rate = 0.001, float lambda1 = 0.0, float lambda2 = 0.0, float dropout = 0.5, unsigned int minibatch_size = 32);

        virtual ~CNN();

        CNN& operator= (CNN& other);
        CNN& operator= (const CNN& other);

    protected:
        void copy(CNN& other);
        void copy(const CNN& other);


    public:
        void forward(Tensor &output, Tensor &input);
        void forward(std::vector<float> &output, std::vector<float> &input);

        void forward(std::vector<std::vector<float>> &output, std::vector<std::vector<float>> &input);

    public:
        void train(Tensor &required_output, Tensor &input);
        void train(std::vector<float> &required_output, std::vector<float> &input);

        void train(std::vector<Tensor> &required_output, std::vector<Tensor> &input);
        void train(std::vector<std::vector<float>> &required_output, std::vector<std::vector<float>> &input);

    public:
        void set_training_mode();
        void unset_training_mode();
        bool is_training_mode();
        void reset();

    public:
        Shape add_layer(std::string layer_type, Shape input_shape = {0, 0, 0});
        std::string asString();

    private:
        void init(Json::Value json_config, Shape input_shape, Shape output_shape);
        std::vector<unsigned int> make_indices(unsigned int count);

        Json::Value default_hyperparameters(float learning_rate = 0.001);


    private:
        Shape m_input_shape, m_output_shape, m_current_input_shape;
        Json::Value m_hyperparameters;

    private:

        Tensor output, required_output, input;

        std::vector<Layer*> layers;

        std::vector<Tensor> l_error, l_output;

        bool training_mode;
        unsigned int minibatch_counter;

        unsigned long int m_total_flops;
        unsigned long int m_total_trainable_parameters;

    private:
        Log network_log;

};

#endif
