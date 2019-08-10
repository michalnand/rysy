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
        void copy_recurrent(RecurrentLayer &other);
        void copy_recurrent(const RecurrentLayer &other);

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights);

        void save(std::string file_name_prefix);
        void load(std::string file_name_prefix);

        bool has_weights() { return true;};

        std::string asString();

    public:
        void reset();

    protected:
        void init_recurrent();

    protected:
        float learning_rate, lambda1, lambda2, gradient_clip;

        Tensor w, bias;
        Tensor w_grad, m, v;

        unsigned int time_step_idx;
        unsigned int time_sequence_length;

        std::vector<Tensor> block_input, block_output;
        std::vector<Tensor> h, h_error;
        Tensor block_error, error_block_back;

};

#endif
