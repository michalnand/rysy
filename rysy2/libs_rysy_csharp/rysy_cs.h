#ifndef _RYSY_PYTHON_H_
#define _RYSY_PYTHON_H_

#include <string>
#include <vector>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/value.h>

#include <image_load.h>
#include <image_save.h>
#include <json_config.h>
#include <log.h>
#include <timer.h>

#include <batch.h>
#include <dataset_images.h>
#include <dataset_interface.h>
#include <dataset_mnist.h>

#include <classification_compare.h>
#include <classification_experiment.h>

#include <cnn.h>
#include <config.h>
#include <shape.h>
#include <tensor.h>

#include <dqn.h>
#include <dqna.h>
#include <experience_replay_buffer.h>

typedef std::vector<char> VectorChar;
typedef std::vector<unsigned char> VectorUnsignedChar;
typedef std::vector<int> VectorInt;
typedef std::vector<unsigned int> VectorUnsignedInt;
typedef std::vector<float> VectorFloat;
typedef std::vector<double> VectorDouble;

typedef std::vector<std::vector<int>> MatrixInt;
typedef std::vector<std::vector<unsigned int>> MatrixUnsignedInt;

typedef std::vector<std::vector<float>> MatrixFloat;
typedef std::vector<std::vector<double>> MatrixDouble;

typedef std::string String;


std::vector<float> VectorFloatCreate(unsigned int size = 0);


#endif
