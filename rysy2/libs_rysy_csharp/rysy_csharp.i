/* File : example_lib.i */
%module rysy


%include <std_vector.i>
%include <std_string.i>

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

#include <classification_compare.h>
#include <classification_experiment.h>

#include <shape.h>
#include <tensor.h>
#include <cnn.h>
#include <cnn_visualisation.h>

#include <experience_replay_buffer.h>
#include <dqn.h>
#include <dqna.h>

#include <layer_export.h>
#include <embedded_network_test.h>
#include <embedded_network_export.h>

#include <rysy_cs.h>
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

%include <classification_compare.h>
%include <classification_experiment.h>

%include <shape.h>
%include <tensor.h>
%include <cnn.h>
%include <cnn_visualisation.h>

%include <experience_replay_buffer.h>
%include <dqn.h>
%include <dqna.h>

%include <layer_export.h>
%include <embedded_network_test.h>
%include <embedded_network_export.h>

%include <rysy_cs.h>
