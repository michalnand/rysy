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

        CNN(std::string network_config_file_name, Shape input_shape = {0, 0, 0}, Shape output_shape = {0, 0, 0});
        CNN(Json::Value json_config, Shape input_shape = {0, 0, 0}, Shape output_shape = {0, 0, 0});
        CNN(Shape input_shape, Shape output_shape, float learning_rate = 0.001, float lambda1 = 0.000001, float lambda2 = 0.000001, float gradient_clip = 10.0, float dropout = 0.5, unsigned int minibatch_size = 32);

        virtual ~CNN();

        CNN& operator= (CNN& other);
        CNN& operator= (const CNN& other);

    protected:
        void copy(CNN& other);
        void copy(const CNN& other);

    public:
        Shape get_input_shape();
        Shape get_output_shape();


    public:
        void forward(Tensor &output, Tensor &input);
        void forward(std::vector<float> &output, std::vector<float> &input);

        void forward(std::vector<std::vector<float>> &output, std::vector<std::vector<float>> &input);

    public:
        void train(Tensor &required_output, Tensor &input);
        void train(std::vector<float> &required_output, std::vector<float> &input);

        void train(std::vector<Tensor> &required_output, std::vector<Tensor> &input, unsigned int epoch_count = 1, bool verbose = true);
        void train(std::vector<std::vector<float>> &required_output, std::vector<std::vector<float>> &input, unsigned int epoch_count = 1, bool verbose = true);

        void train_from_gradient(std::vector<float> &gradient);
        void train_from_gradient(Tensor &gradient);
        Tensor& get_gradient_back();

        std::vector<float> kernel_visualisation(unsigned int layer, unsigned int kernel);
        void kernel_visualisation(std::string image_path);

        void activity_visualisation(std::string image_path, std::vector<float> &input_);

        std::vector<std::vector<float>> heatmap_compute(std::vector<float> &input_);
        void heatmap_visualisation(std::string result_path, std::vector<float> &input_);

    public:
        void set_training_mode();
        void unset_training_mode();
        bool is_training_mode();
        void reset();

    public:
        Shape add_layer(std::string layer_type, Shape shape = {0, 0, 0}, std::string weights_file_name_prefix = "");
        std::string asString();
        void print();

    public:
        void save(std::string path);
        void load_weights(std::string file_name_prefix);

    private:
        void init(Json::Value json_config, Shape input_shape, Shape output_shape);
        std::vector<unsigned int> make_indices(unsigned int count);

        Json::Value default_hyperparameters(float learning_rate = 0.001);

    public:
        unsigned int get_layers_count();
        unsigned int get_layer_output_size();
        Tensor& get_layer_output(unsigned int layer_idx);
        bool get_layer_weights_flag(unsigned int layer_idx);
        bool get_layer_activation_flag(unsigned int layer_idx);

    private:
        Shape m_input_shape, m_output_shape, m_current_input_shape;
        Json::Value m_hyperparameters;
        Json::Value m_parameters;

    private:

        Tensor output, required_output, input, gradient;

        std::vector<Layer*> layers;

        std::vector<Tensor> l_gradient, l_output;

        bool training_mode;
        unsigned int minibatch_size, minibatch_counter;

        unsigned long int m_total_flops;
        unsigned long int m_total_trainable_parameters;

    private:
        Log network_log;

};

#endif
