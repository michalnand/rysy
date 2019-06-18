#ifndef _LAYER_INTERFACE_H_
#define _LAYER_INTERFACE_H_

#include <nn_struct.h>
#include <json_config.h>

class LayerInterface
{
    public:
        LayerInterface();
        LayerInterface(LayerInterface& other);
        LayerInterface(const LayerInterface& other);

        LayerInterface(Json::Value json, sShape input_shape = {0, 0, 0});

        virtual ~LayerInterface();

        LayerInterface& operator= (LayerInterface& other);
        LayerInterface& operator= (const LayerInterface& other);

    protected:
        void copy_interface(LayerInterface& other);
        void copy_interface(const LayerInterface& other);

    public:
        sShape get_input_shape();
        sShape get_output_shape();
        sShape get_kernel_shape();

        unsigned int get_input_size();
        unsigned int get_output_size();

    public:
        virtual void print();
        virtual void forward(float *output, float *input);


    protected:
        Json::Value json;
        sShape input_shape, kernel_shape, output_shape;
};

#endif
