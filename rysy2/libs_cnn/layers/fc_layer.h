#ifndef _FC_LAYER_H_
#define _FC_LAYER_H_

#include <layers/layer.h>

class FCLayer final: public Layer
{
    public:
        FCLayer();
        FCLayer(FCLayer& other);
        FCLayer(const FCLayer& other);

        FCLayer(Shape input_shape, Json::Value parameters, unsigned int max_time_steps = 1);

        virtual ~FCLayer();

        FCLayer& operator= (FCLayer& other);
        FCLayer& operator= (const FCLayer& other);

    protected:
        void copy_fc(FCLayer &other);
        void copy_fc(const FCLayer &other);

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights);

        void save(std::string file_name_prefix);
        void load(std::string file_name_prefix);

        bool has_weights() { return true;};

    protected:
        void init_fc();

    protected:
        Json::Value m_hyperparameters;
        float learning_rate;

        Tensor w, bias;
        Tensor w_grad, m, v;
};

#endif
