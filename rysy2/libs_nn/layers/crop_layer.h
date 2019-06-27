#ifndef _CROP_LAYER_H_
#define _CROP_LAYER_H_

#include <layers/layer.h>

class CropLayer final: public Layer
{
    public:
        CropLayer();
        CropLayer(CropLayer& other);
        CropLayer(const CropLayer& other);

        CropLayer(Shape input_shape, Json::Value parameters);

        virtual ~CropLayer();

        CropLayer& operator= (CropLayer& other);
        CropLayer& operator= (const CropLayer& other);

        std::string asString();

    public:
        void forward(Tensor &output, Tensor &input);
        void backward(Tensor &error_back, Tensor &error, Tensor &input, Tensor &output, bool update_weights);


    protected:
        void init_crop_layer();
};

#endif
