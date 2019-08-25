#ifndef _FIR_LAYER_H_
#define _FIR_LAYER_H_

#include <layers/layer.h>

class FirLayer final: public Layer
{
    public:
        FirLayer();
        FirLayer(FirLayer& other);
        FirLayer(const FirLayer& other);

        FirLayer(Shape input_shape, Json::Value parameters);

        virtual ~FirLayer();

        FirLayer& operator= (FirLayer& other);
        FirLayer& operator= (const FirLayer& other);

    protected:
        void copy_fir_layer(FirLayer &other);
        void copy_fir_layer(const FirLayer &other);

    public:
        void reset();

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights, bool update_bias = true);

        std::string asString();

    protected:
        void init_fir_layer();

    protected:
        unsigned int time_sequence_length;
        std::vector<Tensor> h;
        std::vector<Tensor> error_h;

};

#endif
