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

    private:
        void init(Json::Value json_config, Shape input_shape, Shape output_shape);



    private:
        Shape m_input_shape, m_output_shape;
        Json::Value m_hyperparameters;

    private:

        Tensor output, required_output, input;

        std::vector<Layer> layers;

        std::vector<Tensor> l_error, l_output;

    private:
        Log network_log;

};

#endif
