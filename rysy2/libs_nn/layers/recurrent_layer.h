#ifndef _RECURRENT_LAYER_H_
#define _RECURRENT_LAYER_H_

#include <layers/layer.h>

class RecurrentLayer final: public Layer
{
    public:
        RecurrentLayer();
        RecurrentLayer(RecurrentLayer& other);
        RecurrentLayer(const RecurrentLayer& other);

        RecurrentLayer(Shape input_shape, Json::Value parameters);

        virtual ~RecurrentLayer();

        RecurrentLayer& operator= (RecurrentLayer& other);
        RecurrentLayer& operator= (const RecurrentLayer& other);

    protected:
        void copy_recurrent_layer(RecurrentLayer &other);
        void copy_recurrent_layer(const RecurrentLayer &other);

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
        void init_recurrent_layer();

    protected:
        unsigned int time_sequence_length;

        std::vector<Tensor> h, h_error;
        Tensor activation_error_back, fc_error_back;
        Tensor fc_input, fc_output;


        float learning_rate, lambda1, lambda2, gradient_clip;
        Tensor w, bias;
        Tensor w_grad, m, v;


};

#endif
