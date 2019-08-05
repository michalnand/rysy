#ifndef _RECURRENT_LAYER_H_
#define _RECURRENT_LAYER_H_

#include <layers/layer.h>
#include <weights.h>

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
        void copy_rl(RecurrentLayer &other);
        void copy_rl(const RecurrentLayer &other);

    public:
        void reset();

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights);

        void save(std::string file_name_prefix);
        void load(std::string file_name_prefix);

        bool has_weights() { return true;};

        std::string asString();

    protected:
        void init_rl();

    protected:
        float learning_rate, lambda1, lambda2, gradient_clip;

        unsigned int time_step_idx, time_sequence_length;

        Weights wx;
        Weights wh;

        Tensor bias;
        
        std::vector<Tensor> h, error_h;
};

#endif
