#include <embedded_network_export.h>
#include <fstream>
#include <iostream>
#include <math.h>

#include <log.h>
#include <layer_export.h>

EmbeddedNetworkExport::EmbeddedNetworkExport(std::string trained_config_file_name)
{
  JsonConfig json(trained_config_file_name);

  json_parameters = json.result;
}

EmbeddedNetworkExport::~EmbeddedNetworkExport()
{

}

int EmbeddedNetworkExport::process(std::string export_path, std::string network_prefix)
{
  std::string network_config_cpp;


  network_config_cpp+= "\n\n";
  network_config_cpp+= network_prefix+ "::" + network_prefix + "()\n";
  network_config_cpp+= "\t\t\t:EmbeddedNet()\n";
  network_config_cpp+= "{\n";

  sEmbeddedNetShape input_shape;

  input_shape.w = json_parameters["input_shape"][0].asInt();
  input_shape.h = json_parameters["input_shape"][1].asInt();
  input_shape.d = json_parameters["input_shape"][2].asInt();

  sEmbeddedNetShape output_shape;

  output_shape.w = json_parameters["output_shape"][0].asInt();
  output_shape.h = json_parameters["output_shape"][1].asInt();
  output_shape.d = json_parameters["output_shape"][2].asInt();

  sEmbeddedNetShape shape;

  network_config_cpp+= "\t\tinput_shape.w = " + std::to_string(input_shape.w) + ";\n";
  network_config_cpp+= "\t\tinput_shape.h = " + std::to_string(input_shape.h) + ";\n";
  network_config_cpp+= "\t\tinput_shape.d = " + std::to_string(input_shape.d) + ";\n";
  network_config_cpp+= "\n";

  network_config_cpp+= "\t\toutput_shape.w = " + std::to_string(output_shape.w) + ";\n";
  network_config_cpp+= "\t\toutput_shape.h = " + std::to_string(output_shape.h) + ";\n";
  network_config_cpp+= "\t\toutput_shape.d = " + std::to_string(output_shape.d) + ";\n";
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
      shape.w = layer["shape"][0].asInt();
      shape.h = layer["shape"][1].asInt();
      shape.d = layer["shape"][2].asInt();

      output_shape.w = 1;
      output_shape.h = 1;
      output_shape.d = shape.d;

      std::string layer_prefix = "layer_" + std::to_string(layer_idx);
      LayerExport layer_export(export_path, layer, layer_prefix, input_shape, output_shape);

      network_config_cpp_headers+= "#include \"" + layer_prefix +".h\"\n";

      std::string s_shape = layer_prefix + "_shape";
      std::string s_input_shape  = layer_prefix + "_input_shape";
      std::string s_output_shape = layer_prefix + "_output_shape";
      std::string s_weight = layer_prefix + "_weights";
      std::string s_bias = layer_prefix + "_bias";
      std::string s_weight_range = layer_prefix + "_weights_range";
      std::string s_bias_range = layer_prefix + "_bias_range";


      network_config_cpp+= "\t\tlayers[" + std::to_string(layer_idx)+"] = ";
      network_config_cpp+= "new EmbeddedNetFcLayer(" +
                            s_shape + "," + s_input_shape + "," + s_output_shape + "," +
                            s_weight + "," +
                            s_bias + "," +
                            s_weight_range + "," +
                            s_bias_range+");";

      network_config_cpp+= "\n";

      layer_idx++;
      input_shape = output_shape;
    }


    if (layer["type"].asString() == "convolution")
    {
      shape.w = layer["shape"][0].asInt();
      shape.h = layer["shape"][1].asInt();
      shape.d = layer["shape"][2].asInt();

      output_shape.w = input_shape.w;
      output_shape.h = input_shape.h;
      output_shape.d = shape.d;

      std::string layer_prefix = "layer_" + std::to_string(layer_idx);
      LayerExport layer_export(export_path, layer, layer_prefix, input_shape, output_shape);

      network_config_cpp_headers+= "#include \"" + layer_prefix +".h\"\n";

      std::string s_shape = layer_prefix + "_shape";
      std::string s_input_shape  = layer_prefix + "_input_shape";
      std::string s_output_shape = layer_prefix + "_output_shape";
      std::string s_weight = layer_prefix + "_weights";
      std::string s_bias = layer_prefix + "_bias";
      std::string s_weight_range = layer_prefix + "_weights_range";
      std::string s_bias_range = layer_prefix + "_bias_range";


      network_config_cpp+= "\t\tlayers[" + std::to_string(layer_idx)+"] = ";
      network_config_cpp+= "new EmbeddedNetConvolutionLayer(" +
                            s_shape + "," + s_input_shape + "," + s_output_shape + "," +
                            s_weight + "," +
                            s_bias + "," +
                            s_weight_range + "," +
                            s_bias_range+");";

      network_config_cpp+= "\n";

      layer_idx++;
      input_shape = output_shape;
    }

    if ( (layer["type"].asString() == "dense convolution") || (layer["type"].asString() == "dense_convolution"))
    {
      shape.w = layer["shape"][0].asInt();
      shape.h = layer["shape"][1].asInt();
      shape.d = layer["shape"][2].asInt();

      output_shape.w = input_shape.w;
      output_shape.h = input_shape.h;
      output_shape.d = shape.d + input_shape.d;

      std::string layer_prefix = "layer_" + std::to_string(layer_idx);
      LayerExport layer_export(export_path, layer, layer_prefix, input_shape, output_shape);

      network_config_cpp_headers+= "#include \"" + layer_prefix +".h\"\n";

      std::string s_shape = layer_prefix + "_shape";
      std::string s_input_shape  = layer_prefix + "_input_shape";
      std::string s_output_shape = layer_prefix + "_output_shape";
      std::string s_weight = layer_prefix + "_weights";
      std::string s_bias = layer_prefix + "_bias";
      std::string s_weight_range = layer_prefix + "_weights_range";
      std::string s_bias_range = layer_prefix + "_bias_range";


      network_config_cpp+= "\t\tlayers[" + std::to_string(layer_idx)+"] = ";
      network_config_cpp+= "new EmbeddedNetDenseConvolutionLayer(" +
                            s_shape + "," + s_input_shape + "," + s_output_shape + "," +
                            s_weight + "," +
                            s_bias + "," +
                            s_weight_range + "," +
                            s_bias_range+");";

      network_config_cpp+= "\n";

      layer_idx++;
      input_shape = output_shape;
    }

    if (layer["type"].asString() == "output")
    {
      output_shape.w = json_parameters["output_shape"][0].asInt();
      output_shape.h = json_parameters["output_shape"][1].asInt();
      output_shape.d = json_parameters["output_shape"][2].asInt();

      shape = output_shape;

      std::string layer_prefix = "layer_" + std::to_string(layer_idx);
      LayerExport layer_export(export_path, layer, layer_prefix, input_shape, output_shape);

      network_config_cpp_headers+= "#include \"" + layer_prefix +".h\"\n";

      std::string s_shape = layer_prefix + "_shape";
      std::string s_input_shape  = layer_prefix + "_input_shape";
      std::string s_output_shape = layer_prefix + "_output_shape";
      std::string s_weight = layer_prefix + "_weights";
      std::string s_bias = layer_prefix + "_bias";
      std::string s_weight_range = layer_prefix + "_weights_range";
      std::string s_bias_range = layer_prefix + "_bias_range";


      network_config_cpp+= "\t\tlayers[" + std::to_string(layer_idx)+"] = ";
      network_config_cpp+= "new EmbeddedNetFcLayer(" +
                            s_shape + "," + s_input_shape + "," + s_output_shape + "," +
                            s_weight + "," +
                            s_bias + "," +
                            s_weight_range + "," +
                            s_bias_range+");";

      network_config_cpp+= "\n";


      layer_idx++;
      input_shape = output_shape;
    }

    if ((layer["type"].asString() == "max pooling")||(layer["type"].asString() == "max_pooling"))
    {
      shape.w = layer["shape"][0].asInt();
      shape.h = layer["shape"][1].asInt();
      shape.d = layer["shape"][2].asInt();

      output_shape.w = layer["output_shape"][0].asInt();
      output_shape.h = layer["output_shape"][1].asInt();
      output_shape.d = layer["output_shape"][2].asInt();

      std::string layer_prefix = "layer_" + std::to_string(layer_idx);
      LayerExport layer_export(export_path, layer, layer_prefix, input_shape, output_shape);

      network_config_cpp_headers+= "#include \"" + layer_prefix +".h\"\n";

      std::string s_shape = layer_prefix + "_shape";
      std::string s_input_shape  = layer_prefix + "_input_shape";
      std::string s_output_shape = layer_prefix + "_output_shape";

      network_config_cpp+= "\t\tlayers[" + std::to_string(layer_idx)+"] = ";
      network_config_cpp+= "new EmbeddedNetMaxPoolingLayer(" + s_shape + "," + s_input_shape + "," + s_output_shape + ");";
      network_config_cpp+= "\n";

      layer_idx++;
      input_shape = output_shape;
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
  network_config_h+= "#include <EmbeddedNet.h>\n";
  network_config_h+= "\n";

  network_config_h+= "class " + network_prefix + ": public EmbeddedNet\n";
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
