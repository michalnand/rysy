#ifndef _GRU_LAYER_H_
#define _GRU_LAYER_H_

#include <layers/layer.h>
#include <weights.h>

class GRULayer final: public Layer
{
    public:
        GRULayer();
        GRULayer(GRULayer& other);
        GRULayer(const GRULayer& other);

        GRULayer(Shape input_shape, Json::Value parameters);

        virtual ~GRULayer();

        GRULayer& operator= (GRULayer& other);
        GRULayer& operator= (const GRULayer& other);

    protected:
        void copy_gru_layer(GRULayer &other);
        void copy_gru_layer(const GRULayer &other);

    public:
        void reset();

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights_);

        std::string asString();

        bool has_weights() { return true;};

    protected:
        void init_gru_layer();

    protected:
        float learning_rate, lambda1, lambda2, gradient_clip;

        unsigned int time_step_idx;
        unsigned int time_sequence_length;


        Weights control_weights, update_weights;
        Tensor control_bias, update_bias;

        Tensor block_input;

        std::vector<Tensor> h;
        std::vector<Tensor> control_output, update_output;
        std::vector<Tensor> h_error;

        Tensor control_h_error_back, update_h_error_back;
        Tensor control_error_back, update_error_back;
        Tensor tmp_error, tmp_error_h;

};

#endif
