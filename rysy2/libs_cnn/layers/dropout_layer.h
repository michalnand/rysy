#ifndef _DROPOUT_LAYER_H_
#define _DROPOUT_LAYER_H_

#include <layers/layer.h>

class DropoutLayer final: public Layer
{
    public:
        DropoutLayer();
        DropoutLayer(DropoutLayer& other);
        DropoutLayer(const DropoutLayer& other);

        DropoutLayer(Shape input_shape, Json::Value parameters);

        virtual ~DropoutLayer();

        DropoutLayer& operator= (DropoutLayer& other);
        DropoutLayer& operator= (const DropoutLayer& other);

        std::string asString();

    protected:
        void copy_dropout(DropoutLayer &other);
        void copy_dropout(const DropoutLayer &other);


    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights);


    protected:
        void init_dropout_layer();

    private:
        float m_dropout_level;
        Tensor noise;
};

#endif
