#ifndef _HIGHWAY_BLOCK_LAYER_H_
#define _HIGHWAY_BLOCK_LAYER_H_

#include <layers/layer.h>

class HighwayBlockLayer final: public Layer
{
    public:
        HighwayBlockLayer();
        HighwayBlockLayer(HighwayBlockLayer& other);
        HighwayBlockLayer(const HighwayBlockLayer& other);

        HighwayBlockLayer(Shape input_shape, Json::Value parameters);

        virtual ~HighwayBlockLayer();

        HighwayBlockLayer& operator= (HighwayBlockLayer& other);
        HighwayBlockLayer& operator= (const HighwayBlockLayer& other);
        
    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights);

        std::string asString();

    protected:
        void init_highway_block_layer();
};

#endif
