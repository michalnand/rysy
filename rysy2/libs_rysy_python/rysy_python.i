/* File : example_lib.i */
%module rysy

%include <std_vector.i>
%include <std_string.i>

%template(VectorChar) std::vector<char>;
%template(VectorUnsignedChar) std::vector<unsigned char>;
%template(VectorInt) std::vector<int>;
%template(VectorUnsignedInt) std::vector<unsigned int>;
%template(VectorFloat) std::vector<float>;
%template(VectorDouble) std::vector<double>;

%template(MatrixInt) std::vector<std::vector<int>>;
%template(MatrixUnsignedInt) std::vector<std::vector<unsigned int>>;

%template(MatrixFloat) std::vector<std::vector<float>>;
%template(MatrixDouble) std::vector<std::vector<double>>;


%{
#include <string>
#include <vector>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/value.h>

#include <config.h>

#include <image_load.h>
#include <image_save.h>
#include <json_config.h>
#include <log.h>
#include <timer.h>

#include <batch.h>
#include <dataset_interface.h>
#include <dataset_images.h>
#include <dataset_mnist.h>
#include <dataset_mnist_rnn.h>
#include <dataset_test.h>

#include <classification_compare.h>
#include <classification_experiment.h>

#include <shape.h>
#include <tensor.h>
#include <cnn.h>
#include <rnn.h>
#include <cnn_visualisation.h>

#include <experience_replay_buffer.h>
#include <dqn.h>
#include <dqrn.h>
#include <dqna.h>
#include <icm.h>
#include <dqn_curiosity.h>
#include <stacked_state.h>

#include <layer_export.h>
#include <embedded_network_test.h>
#include <embedded_network_export.h>

%}


%include <config.h>

%include <image_load.h>
%include <image_save.h>
%include <json_config.h>
%include <log.h>
%include <timer.h>

%include <batch.h>
%include <dataset_interface.h>
%include <dataset_images.h>
%include <dataset_mnist.h>
%include <dataset_mnist_rnn.h>
%include <dataset_test.h>

%include <classification_compare.h>
%include <classification_experiment.h>

%include <shape.h>
%include <tensor.h>
%include <cnn.h>
%include <rnn.h>
%include <cnn_visualisation.h>

%include <experience_replay_buffer.h>
%include <dqn.h>
%include <dqrn.h>
%include <dqna.h>
%include <icm.h>
%include <dqn_curiosity.h>
%include <stacked_state.h>

%include <layer_export.h>
%include <embedded_network_test.h>
%include <embedded_network_export.h>
