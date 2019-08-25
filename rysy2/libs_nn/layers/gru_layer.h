#ifndef _GRU_LAYER_H_
#define _GRU_LAYER_H_

#include <layers/layer.h>

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
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias = true);

        void save(std::string file_name_prefix);
        void load(std::string file_name_prefix);
        std::string asString();

        bool has_weights() { return true;};

    protected:
        void init_gru_layer();

    protected:
        unsigned int time_sequence_length;

        std::vector<Tensor> h, h_error;


        Tensor fc_input;
        std::vector<Tensor> fc_output_control, fc_output_update;

        float learning_rate, lambda1, lambda2, gradient_clip;

        Tensor w_control, bias_control;
        Tensor w_grad_control, m_control, v_control;
        Tensor w_update, bias_update;
        Tensor w_grad_update, m_update, v_update;

        Tensor control_error_back, update_error_back;
        Tensor gate_control_error_back, gate_h_error_back, gate_update_error_back;
};

#endif
