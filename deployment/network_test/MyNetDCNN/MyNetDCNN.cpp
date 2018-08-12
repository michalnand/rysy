#include "MyNetDCNN.h"

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
#include "layer_13.h"
#include "layer_14.h"
#include "layer_15.h"
#include "layer_16.h"


MyNetDCNN::MyNetDCNN()
			:NeuralNetwork()
{
		input_geometry.w = 28;
		input_geometry.h = 28;
		input_geometry.d = 1;

		output_geometry.w = 1;
		output_geometry.h = 1;
		output_geometry.d = 10;

		layers[0] = new NetConvolutionLayer(layer_0_kernel_geometry,layer_0_input_geometry,layer_0_output_geometry,layer_0_weights,layer_0_bias,layer_0_weights_range,layer_0_bias_range);
		layers[1] = new NetReluLayer(layer_1_kernel_geometry,layer_1_input_geometry,layer_1_output_geometry);
		layers[2] = new NetMaxPoolingLayer(layer_2_kernel_geometry,layer_2_input_geometry,layer_2_output_geometry);
		layers[3] = new NetDenseConvolutionLayer(layer_3_kernel_geometry,layer_3_input_geometry,layer_3_output_geometry,layer_3_weights,layer_3_bias,layer_3_weights_range,layer_3_bias_range);
		layers[4] = new NetReluLayer(layer_4_kernel_geometry,layer_4_input_geometry,layer_4_output_geometry);
		layers[5] = new NetDenseConvolutionLayer(layer_5_kernel_geometry,layer_5_input_geometry,layer_5_output_geometry,layer_5_weights,layer_5_bias,layer_5_weights_range,layer_5_bias_range);
		layers[6] = new NetReluLayer(layer_6_kernel_geometry,layer_6_input_geometry,layer_6_output_geometry);
		layers[7] = new NetMaxPoolingLayer(layer_7_kernel_geometry,layer_7_input_geometry,layer_7_output_geometry);
		layers[8] = new NetDenseConvolutionLayer(layer_8_kernel_geometry,layer_8_input_geometry,layer_8_output_geometry,layer_8_weights,layer_8_bias,layer_8_weights_range,layer_8_bias_range);
		layers[9] = new NetReluLayer(layer_9_kernel_geometry,layer_9_input_geometry,layer_9_output_geometry);
		layers[10] = new NetDenseConvolutionLayer(layer_10_kernel_geometry,layer_10_input_geometry,layer_10_output_geometry,layer_10_weights,layer_10_bias,layer_10_weights_range,layer_10_bias_range);
		layers[11] = new NetReluLayer(layer_11_kernel_geometry,layer_11_input_geometry,layer_11_output_geometry);
		layers[12] = new NetDenseConvolutionLayer(layer_12_kernel_geometry,layer_12_input_geometry,layer_12_output_geometry,layer_12_weights,layer_12_bias,layer_12_weights_range,layer_12_bias_range);
		layers[13] = new NetReluLayer(layer_13_kernel_geometry,layer_13_input_geometry,layer_13_output_geometry);
		layers[14] = new NetDenseConvolutionLayer(layer_14_kernel_geometry,layer_14_input_geometry,layer_14_output_geometry,layer_14_weights,layer_14_bias,layer_14_weights_range,layer_14_bias_range);
		layers[15] = new NetReluLayer(layer_15_kernel_geometry,layer_15_input_geometry,layer_15_output_geometry);
		layers[16] = new NetFcLayer(layer_16_kernel_geometry,layer_16_input_geometry,layer_16_output_geometry,layer_16_weights,layer_16_bias,layer_16_weights_range,layer_16_bias_range);

		layers_count = 17;
		allocate_buffer();
}


MyNetDCNN::~MyNetDCNN()
{
		for (unsigned int i = 0; i < layers_count; i++)
		{
			delete layers[i];
			layers[i] = nullptr;
		}
}
