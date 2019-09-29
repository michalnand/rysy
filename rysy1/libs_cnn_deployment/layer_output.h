#ifndef _LAYER_OUTPUT_H_
#define _LAYER_OUTPUT_H_

#include <layer_interface.h>

class LayerOutput: public LayerInterface
{
    public:
        LayerOutput();
        LayerOutput(LayerOutput& other);
        LayerOutput(const LayerOutput& other);

        LayerOutput(Json::Value json, sGeometry input_geometry = {0, 0, 0});

        virtual ~LayerOutput();

        LayerOutput& operator= (LayerOutput& other);
        LayerOutput& operator= (const LayerOutput& other);

    protected:
        void copy(LayerOutput& other);
        void copy(const LayerOutput& other);

    public:
        void print();
        void forward(float *output, float *input);

    private:
        float *weights, *bias;

};

#endif
