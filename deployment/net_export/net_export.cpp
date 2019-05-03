#include "net_export.h"
#include <fstream>
#include <iostream>
#include <math.h>

#include <log.h>
#include <layer_export.h>

NetExport::NetExport(std::string trained_config_file_name)
{
  JsonConfig json(trained_config_file_name);

  json_parameters = json.result;
}

NetExport::~NetExport()
{

}

int NetExport::process(std::string export_path, std::string network_prefix)
{
  std::string network_config_cpp;


  network_config_cpp+= "\n\n";
  network_config_cpp+= network_prefix+ "::" + network_prefix + "()\n";
  network_config_cpp+= "\t\t\t:NeuralNetwork()\n";
  network_config_cpp+= "{\n";

  sGeometry input_geometry;

  input_geometry.w = json_parameters["input_geometry"][0].asInt();
  input_geometry.h = json_parameters["input_geometry"][1].asInt();
  input_geometry.d = json_parameters["input_geometry"][2].asInt();

  sGeometry output_geometry;

  output_geometry.w = json_parameters["output_geometry"][0].asInt();
  output_geometry.h = json_parameters["output_geometry"][1].asInt();
  output_geometry.d = json_parameters["output_geometry"][2].asInt();

  sGeometry kernel_geometry;

  network_config_cpp+= "\t\tinput_geometry.w = " + std::to_string(input_geometry.w) + ";\n";
  network_config_cpp+= "\t\tinput_geometry.h = " + std::to_string(input_geometry.h) + ";\n";
  network_config_cpp+= "\t\tinput_geometry.d = " + std::to_string(input_geometry.d) + ";\n";
  network_config_cpp+= "\n";

  network_config_cpp+= "\t\toutput_geometry.w = " + std::to_string(output_geometry.w) + ";\n";
  network_config_cpp+= "\t\toutput_geometry.h = " + std::to_string(output_geometry.h) + ";\n";
  network_config_cpp+= "\t\toutput_geometry.d = " + std::to_string(output_geometry.d) + ";\n";
  network_config_cpp+= "\n";

  unsigned int layer_idx = 0;

  std::string network_config_cpp_headers;

  network_config_cpp_headers+= "#include \"" + network_prefix + ".h\"";
  network_config_cpp_headers+= "\n\n";

  for (unsigned int i = 0; i < json_parameters["layers"].size(); i++)
  {
    auto layer = json_parameters["layers"][i];

    if (layer["type"].asString() == "fc")
    {
      kernel_geometry.w = layer["geometry"][0].asInt();
      kernel_geometry.h = layer["geometry"][1].asInt();
      kernel_geometry.d = layer["geometry"][2].asInt();

      output_geometry.w = 1;
      output_geometry.h = 1;
      output_geometry.d = kernel_geometry.d;

      std::string layer_prefix = "layer_" + std::to_string(layer_idx);
      LayerExport layer_export(export_path, layer, layer_prefix, input_geometry, output_geometry);

      network_config_cpp_headers+= "#include \"" + layer_prefix +".h\"\n";

      std::string s_kernel_geometry = layer_prefix + "_kernel_geometry";
      std::string s_input_geometry  = layer_prefix + "_input_geometry";
      std::string s_output_geometry = layer_prefix + "_output_geometry";
      std::string s_weight = layer_prefix + "_weights";
      std::string s_bias = layer_prefix + "_bias";
      std::string s_weight_range = layer_prefix + "_weights_range";
      std::string s_bias_range = layer_prefix + "_bias_range";


      network_config_cpp+= "\t\tlayers[" + std::to_string(layer_idx)+"] = ";
      network_config_cpp+= "new NetFcLayer(" +
                            s_kernel_geometry + "," + s_input_geometry + "," + s_output_geometry + "," +
                            s_weight + "," +
                            s_bias + "," +
                            s_weight_range + "," +
                            s_bias_range+");";

      network_config_cpp+= "\n";

      layer_idx++;
      input_geometry = output_geometry;
    }

    if ( (layer["type"].asString() == "dense fc") || (layer["type"].asString() == "dense_fc") )
    {
      kernel_geometry.w = layer["geometry"][0].asInt();
      kernel_geometry.h = layer["geometry"][1].asInt();
      kernel_geometry.d = layer["geometry"][2].asInt();

      output_geometry.w = 1;
      output_geometry.h = 1;
      output_geometry.d = kernel_geometry.d + input_geometry.w*input_geometry.h*input_geometry.d;

      std::string layer_prefix = "layer_" + std::to_string(layer_idx);
      LayerExport layer_export(export_path, layer, layer_prefix, input_geometry, output_geometry);

      network_config_cpp_headers+= "#include \"" + layer_prefix +".h\"\n";

      std::string s_kernel_geometry = layer_prefix + "_kernel_geometry";
      std::string s_input_geometry  = layer_prefix + "_input_geometry";
      std::string s_output_geometry = layer_prefix + "_output_geometry";
      std::string s_weight = layer_prefix + "_weights";
      std::string s_bias = layer_prefix + "_bias";
      std::string s_weight_range = layer_prefix + "_weights_range";
      std::string s_bias_range = layer_prefix + "_bias_range";


      network_config_cpp+= "\t\tlayers[" + std::to_string(layer_idx)+"] = ";
      network_config_cpp+= "new NetDenseFcLayer(" +
                            s_kernel_geometry + "," + s_input_geometry + "," + s_output_geometry + "," +
                            s_weight + "," +
                            s_bias + "," +
                            s_weight_range + "," +
                            s_bias_range+");";

      network_config_cpp+= "\n";

      layer_idx++;
      input_geometry = output_geometry;
    }


    if (layer["type"].asString() == "convolution")
    {
      kernel_geometry.w = layer["geometry"][0].asInt();
      kernel_geometry.h = layer["geometry"][1].asInt();
      kernel_geometry.d = layer["geometry"][2].asInt();

      output_geometry.w = input_geometry.w;
      output_geometry.h = input_geometry.h;
      output_geometry.d = kernel_geometry.d;

      std::string layer_prefix = "layer_" + std::to_string(layer_idx);
      LayerExport layer_export(export_path, layer, layer_prefix, input_geometry, output_geometry);

      network_config_cpp_headers+= "#include \"" + layer_prefix +".h\"\n";

      std::string s_kernel_geometry = layer_prefix + "_kernel_geometry";
      std::string s_input_geometry  = layer_prefix + "_input_geometry";
      std::string s_output_geometry = layer_prefix + "_output_geometry";
      std::string s_weight = layer_prefix + "_weights";
      std::string s_bias = layer_prefix + "_bias";
      std::string s_weight_range = layer_prefix + "_weights_range";
      std::string s_bias_range = layer_prefix + "_bias_range";


      network_config_cpp+= "\t\tlayers[" + std::to_string(layer_idx)+"] = ";
      network_config_cpp+= "new NetConvolutionLayer(" +
                            s_kernel_geometry + "," + s_input_geometry + "," + s_output_geometry + "," +
                            s_weight + "," +
                            s_bias + "," +
                            s_weight_range + "," +
                            s_bias_range+");";

      network_config_cpp+= "\n";

      layer_idx++;
      input_geometry = output_geometry;
    }

    if ( (layer["type"].asString() == "dense convolution") || (layer["type"].asString() == "dense_convolution"))
    {
      kernel_geometry.w = layer["geometry"][0].asInt();
      kernel_geometry.h = layer["geometry"][1].asInt();
      kernel_geometry.d = layer["geometry"][2].asInt();

      output_geometry.w = input_geometry.w;
      output_geometry.h = input_geometry.h;
      output_geometry.d = kernel_geometry.d + input_geometry.d;

      std::string layer_prefix = "layer_" + std::to_string(layer_idx);
      LayerExport layer_export(export_path, layer, layer_prefix, input_geometry, output_geometry);

      network_config_cpp_headers+= "#include \"" + layer_prefix +".h\"\n";

      std::string s_kernel_geometry = layer_prefix + "_kernel_geometry";
      std::string s_input_geometry  = layer_prefix + "_input_geometry";
      std::string s_output_geometry = layer_prefix + "_output_geometry";
      std::string s_weight = layer_prefix + "_weights";
      std::string s_bias = layer_prefix + "_bias";
      std::string s_weight_range = layer_prefix + "_weights_range";
      std::string s_bias_range = layer_prefix + "_bias_range";


      network_config_cpp+= "\t\tlayers[" + std::to_string(layer_idx)+"] = ";
      network_config_cpp+= "new NetDenseConvolutionLayer(" +
                            s_kernel_geometry + "," + s_input_geometry + "," + s_output_geometry + "," +
                            s_weight + "," +
                            s_bias + "," +
                            s_weight_range + "," +
                            s_bias_range+");";

      network_config_cpp+= "\n";

      layer_idx++;
      input_geometry = output_geometry;
    }

    if (layer["type"].asString() == "output")
    {
      output_geometry.w = json_parameters["output_geometry"][0].asInt();
      output_geometry.h = json_parameters["output_geometry"][1].asInt();
      output_geometry.d = json_parameters["output_geometry"][2].asInt();

      kernel_geometry = output_geometry;

      std::string layer_prefix = "layer_" + std::to_string(layer_idx);
      LayerExport layer_export(export_path, layer, layer_prefix, input_geometry, output_geometry);

      network_config_cpp_headers+= "#include \"" + layer_prefix +".h\"\n";

      std::string s_kernel_geometry = layer_prefix + "_kernel_geometry";
      std::string s_input_geometry  = layer_prefix + "_input_geometry";
      std::string s_output_geometry = layer_prefix + "_output_geometry";
      std::string s_weight = layer_prefix + "_weights";
      std::string s_bias = layer_prefix + "_bias";
      std::string s_weight_range = layer_prefix + "_weights_range";
      std::string s_bias_range = layer_prefix + "_bias_range";


      network_config_cpp+= "\t\tlayers[" + std::to_string(layer_idx)+"] = ";
      network_config_cpp+= "new NetFcLayer(" +
                            s_kernel_geometry + "," + s_input_geometry + "," + s_output_geometry + "," +
                            s_weight + "," +
                            s_bias + "," +
                            s_weight_range + "," +
                            s_bias_range+");";

      network_config_cpp+= "\n";


      layer_idx++;
      input_geometry = output_geometry;
    }

    /*
    if (layer["type"].asString() == "relu")
    {
      kernel_geometry.w = input_geometry.w;
      kernel_geometry.h = input_geometry.h;
      kernel_geometry.d = input_geometry.d;

      output_geometry = input_geometry;

      std::string layer_prefix = "layer_" + std::to_string(layer_idx);
      LayerExport layer_export(export_path, layer, layer_prefix, input_geometry, output_geometry);

      network_config_cpp_headers+= "#include \"" + layer_prefix +".h\"\n";

      std::string s_kernel_geometry = layer_prefix + "_kernel_geometry";
      std::string s_input_geometry  = layer_prefix + "_input_geometry";
      std::string s_output_geometry = layer_prefix + "_output_geometry";

      network_config_cpp+= "\t\tlayers[" + std::to_string(layer_idx)+"] = ";
      network_config_cpp+= "new NetReluLayer(" + s_kernel_geometry + "," + s_input_geometry + "," + s_output_geometry + ");";
      network_config_cpp+= "\n";

      layer_idx++;
      input_geometry = output_geometry;
    }
    */

    if ((layer["type"].asString() == "max pooling")||(layer["type"].asString() == "max_pooling"))
    {
      kernel_geometry.w = layer["geometry"][0].asInt();
      kernel_geometry.h = layer["geometry"][1].asInt();
      kernel_geometry.d = layer["geometry"][2].asInt();

      output_geometry.w = input_geometry.w/kernel_geometry.w;
      output_geometry.h = input_geometry.h/kernel_geometry.h;
      output_geometry.d = input_geometry.d;

      std::string layer_prefix = "layer_" + std::to_string(layer_idx);
      LayerExport layer_export(export_path, layer, layer_prefix, input_geometry, output_geometry);

      network_config_cpp_headers+= "#include \"" + layer_prefix +".h\"\n";

      std::string s_kernel_geometry = layer_prefix + "_kernel_geometry";
      std::string s_input_geometry  = layer_prefix + "_input_geometry";
      std::string s_output_geometry = layer_prefix + "_output_geometry";

      network_config_cpp+= "\t\tlayers[" + std::to_string(layer_idx)+"] = ";
      network_config_cpp+= "new NetMaxPoolingLayer(" + s_kernel_geometry + "," + s_input_geometry + "," + s_output_geometry + ");";
      network_config_cpp+= "\n";

      layer_idx++;
      input_geometry = output_geometry;
    }

  }

  network_config_cpp+= "\n";
  network_config_cpp+= "\t\tlayers_count = " + std::to_string(layer_idx) + ";\n";
  network_config_cpp+= "\t\tallocate_buffer();\n";
  network_config_cpp+= "}\n";


  network_config_cpp+= "\n\n";
  network_config_cpp+= network_prefix+ "::~" + network_prefix + "()\n";
  network_config_cpp+= "{\n";
  network_config_cpp+= "\t\tfor (unsigned int i = 0; i < layers_count; i++)\n";
  network_config_cpp+= "\t\t{\n";
  network_config_cpp+= "\t\t\tdelete layers[i];\n";
  network_config_cpp+= "\t\t\tlayers[i] = nullptr;\n";
  network_config_cpp+= "\t\t}\n";
  network_config_cpp+= "}\n";


  Log network_config_file_cpp(export_path+network_prefix+".cpp");
  network_config_file_cpp << network_config_cpp_headers;
  network_config_file_cpp << network_config_cpp;


  std::string network_config_h;

  network_config_h+= "#ifndef _NETWORK_" + network_prefix + "_H_\n";
  network_config_h+= "#define _NETWORK_" + network_prefix + "_H_\n";
  network_config_h+= "\n";
  network_config_h+= "#include <NeuralNetwork.h>\n";
  network_config_h+= "\n";

  network_config_h+= "class " + network_prefix + ": public NeuralNetwork\n";
  network_config_h+= "{\n";

  network_config_h+= "\tpublic:\n";
  network_config_h+= "\t\t" + network_prefix + "();\n";
  network_config_h+= "\t\tvirtual ~" + network_prefix + "();\n";

  network_config_h+= "};\n\n";
  network_config_h+= "#endif\n";

  std::cout << "eporting cpp config as " << export_path+network_prefix+".cpp\n";
  std::cout << "eporting h   config as " << export_path+network_prefix+".h\n";

  Log network_config_file_h(export_path+network_prefix+".h");
  network_config_file_h << network_config_h;

  return 0;
}
