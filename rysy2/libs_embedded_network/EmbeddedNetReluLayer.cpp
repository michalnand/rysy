#include <EmbeddedNetReluLayer.h>


EmbeddedNetReluLayer::EmbeddedNetReluLayer(sEmbeddedNetShape kernel_shape, sEmbeddedNetShape input_shape, sEmbeddedNetShape output_shape)
             :EmbeddedNetLayer()
{
    m_kernel_shape = kernel_shape;
    m_input_shape  = input_shape;
    m_output_shape = output_shape;

    size = m_input_shape.w*m_input_shape.h*m_input_shape.d;
}

EmbeddedNetReluLayer::~EmbeddedNetReluLayer()
{

}

void EmbeddedNetReluLayer::forward(nn_layer_t *output, nn_layer_t *input)
{
    for (unsigned int i = 0; i < size; i++)
    {
        if (input[i] < 0)
            output[i] = 0;
        else
            output[i] = input[i];
    }
}
