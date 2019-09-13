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

        std::string asString();

        bool is_activation() {return true;}

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias = true);


    protected:
        void init_spatial_attention_layer();
};

#endif
