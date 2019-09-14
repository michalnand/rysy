#ifndef _SPATIAL_ATTENTION_LAYER_H_
#define _SPATIAL_ATTENTION_LAYER_H_

#include <layers/layer.h>

class SpatialAttentionLayer final: public Layer
{
    public:
        SpatialAttentionLayer();
        SpatialAttentionLayer(SpatialAttentionLayer& other);
        SpatialAttentionLayer(const SpatialAttentionLayer& other);

        SpatialAttentionLayer(Shape input_shape, Json::Value parameters);

        virtual ~SpatialAttentionLayer();

        SpatialAttentionLayer& operator= (SpatialAttentionLayer& other);
        SpatialAttentionLayer& operator= (const SpatialAttentionLayer& other);

        void copy_spatial_attention(SpatialAttentionLayer &other);
        void copy_spatial_attention(const SpatialAttentionLayer &other);


        std::string asString();

        bool is_activation() {return true;}
        bool has_weights() { return true;}

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias = true);

        void save(std::string file_name_prefix);
        void load(std::string file_name_prefix);

    protected:
        void init_spatial_attention_layer();

    protected:
        float learning_rate, lambda1, lambda2, gradient_clip;

        Tensor w, bias;
        Tensor w_grad, m, v;

        Shape m_kernel_shape;

        Tensor input_attention;
        Tensor error_back_attention;
        Tensor error_back_conv;
};

#endif
