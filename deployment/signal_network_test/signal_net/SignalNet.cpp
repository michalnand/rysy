#include "SignalNet.h"

#include "layer_0.h"
#include "layer_1.h"
#include "layer_2.h"
#include "layer_3.h"
#include "layer_4.h"
#include "layer_5.h"
#include "layer_6.h"
#include "layer_7.h"
#include "layer_8.h"
#include "layer_9.h"
#include "layer_10.h"
#include "layer_11.h"
#include "layer_12.h"


SignalNet::SignalNet()
			:NeuralNetwork()
{
		input_geometry.w = 128;
		input_geometry.h = 1;
		input_geometry.d = 1;

		output_geometry.w = 1;
		output_geometry.h = 1;
		output_geometry.d = 10;

		layers[0] = new NetConvolutionLayer(layer_0_kernel_geometry,layer_0_input_geometry,layer_0_output_geometry,layer_0_weights,layer_0_bias,layer_0_weights_range,layer_0_bias_range);
		layers[1] = new NetReluLayer(layer_1_kernel_geometry,layer_1_input_geometry,layer_1_output_geometry);
		layers[2] = new NetMaxPoolingLayer(layer_2_kernel_geometry,layer_2_input_geometry,layer_2_output_geometry);
		layers[3] = new NetConvolutionLayer(layer_3_kernel_geometry,layer_3_input_geometry,layer_3_output_geometry,layer_3_weights,layer_3_bias,layer_3_weights_range,layer_3_bias_range);
		layers[4] = new NetReluLayer(layer_4_kernel_geometry,layer_4_input_geometry,layer_4_output_geometry);
		layers[5] = new NetMaxPoolingLayer(layer_5_kernel_geometry,layer_5_input_geometry,layer_5_output_geometry);
		layers[6] = new NetConvolutionLayer(layer_6_kernel_geometry,layer_6_input_geometry,layer_6_output_geometry,layer_6_weights,layer_6_bias,layer_6_weights_range,layer_6_bias_range);
		layers[7] = new NetReluLayer(layer_7_kernel_geometry,layer_7_input_geometry,layer_7_output_geometry);
		layers[8] = new NetMaxPoolingLayer(layer_8_kernel_geometry,layer_8_input_geometry,layer_8_output_geometry);
		layers[9] = new NetConvolutionLayer(layer_9_kernel_geometry,layer_9_input_geometry,layer_9_output_geometry,layer_9_weights,layer_9_bias,layer_9_weights_range,layer_9_bias_range);
		layers[10] = new NetReluLayer(layer_10_kernel_geometry,layer_10_input_geometry,layer_10_output_geometry);
		layers[11] = new NetMaxPoolingLayer(layer_11_kernel_geometry,layer_11_input_geometry,layer_11_output_geometry);
		layers[12] = new NetFcLayer(layer_12_kernel_geometry,layer_12_input_geometry,layer_12_output_geometry,layer_12_weights,layer_12_bias,layer_12_weights_range,layer_12_bias_range);

		layers_count = 13;
		allocate_buffer();
}


SignalNet::~SignalNet()
{
		for (unsigned int i = 0; i < layers_count; i++)
		{
			delete layers[i];
			layers[i] = nullptr;
		}
}
