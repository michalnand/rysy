#ifndef _CONVOLUTION_LAYER_H_
#define _CONVOLUTION_LAYER_H_

#include <layers/layer.h>

class ConvolutionLayer final: public Layer
{
    public:
        ConvolutionLayer();
        ConvolutionLayer(ConvolutionLayer& other);
        ConvolutionLayer(const ConvolutionLayer& other);

        ConvolutionLayer(Shape input_shape, Json::Value parameters);

        virtual ~ConvolutionLayer();

        ConvolutionLayer& operator= (ConvolutionLayer& other);
        ConvolutionLayer& operator= (const ConvolutionLayer& other);

    protected:
        void copy_convolution(ConvolutionLayer &other);
        void copy_convolution(const ConvolutionLayer &other);

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias = true);

        void save(std::string file_name_prefix);
        void load(std::string file_name_prefix);

        bool has_weights() { return true;};

        std::string asString();

    protected:
        void init_convolution();

    protected:
        float learning_rate, lambda1, lambda2, gradient_clip;

        Tensor w, bias;
        Tensor w_grad, m, v;

        Shape m_kernel_shape;
};

#endif
