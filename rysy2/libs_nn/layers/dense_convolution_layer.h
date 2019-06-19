#ifndef _DENSE_CONVOLUTION_LAYER_H_
#define _DENSE_CONVOLUTION_LAYER_H_

#include <layers/layer.h>

class DenseConvolutionLayer final: public Layer
{
    public:
        DenseConvolutionLayer();
        DenseConvolutionLayer(DenseConvolutionLayer& other);
        DenseConvolutionLayer(const DenseConvolutionLayer& other);

        DenseConvolutionLayer(Shape input_shape, Json::Value parameters);

        virtual ~DenseConvolutionLayer();

        DenseConvolutionLayer& operator= (DenseConvolutionLayer& other);
        DenseConvolutionLayer& operator= (const DenseConvolutionLayer& other);

    protected:
        void copy_dense_convolution(DenseConvolutionLayer &other);
        void copy_dense_convolution(const DenseConvolutionLayer &other);

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights);

        void save(std::string file_name_prefix);
        void load(std::string file_name_prefix);

        bool has_weights() { return true;};

        std::string asString();

    protected:
        void init_dense_convolution();

    protected:
        float learning_rate, lambda1, lambda2;

        Tensor w, bias;
        Tensor w_grad, m, v;

        Tensor m_conv_output, m_error_convolution, m_error_direct;

        Shape m_kernel_shape;
};

#endif
