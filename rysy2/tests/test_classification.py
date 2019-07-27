import sys
sys.path.append("../libs_rysy_python")

from rysy import *
import time

#load dataset
dataset_path = "/home/michal/dataset/mnist/"
dataset = DatasetMnist(dataset_path + "train-images-idx3-ubyte",
                            dataset_path + "train-labels-idx1-ubyte",
                            dataset_path + "t10k-images-idx3-ubyte",
                            dataset_path + "t10k-labels-idx1-ubyte")

'''
create example network
3 convolutional layers, dropout and full connected layer as output
C3x3x16 - P2x2 - C3x3x32 - P2x2 - C3x3x32 - FC10
'''

input_shape = dataset.get_input_shape()
output_shape = dataset.get_output_shape()


cnn = CNN(input_shape, output_shape, 0.0025)

cnn.add_layer("convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("max_pooling", Shape(2, 2))

cnn.add_layer("convolution", Shape(3, 3, 32))
cnn.add_layer("elu")
cnn.add_layer("max_pooling", Shape(2, 2))
cnn.add_layer("crop")

cnn.add_layer("convolution", Shape(3, 3, 32))
cnn.add_layer("elu")
cnn.add_layer("crop")
cnn.add_layer("flatten")

cnn.add_layer("dropout")
cnn.add_layer("output")

cnn._print()

time_start = int(round(time.time() * 1000))
#train network - set epoch count
epoch_count = 1
cnn.train(dataset.get_training_output_all(), dataset.get_training_input_all(), epoch_count)


#test network response on whole testing dataset items

compare = ClassificationCompare(dataset.get_classes_count())

nn_output = VectorFloat(dataset.get_classes_count())

#for all testing items
for item_idx in range(0, dataset.get_testing_count()):
    #get network response
    cnn.forward(nn_output, dataset.get_testing_input(item_idx))

    #compare with testing dataset
    compare.add(dataset.get_testing_output(item_idx), nn_output)

    if compare.is_nan_error():
        print("NaN error")

#process computing and print results
compare.compute()

time_stop = int(round(time.time() * 1000))

print(compare.asString())
print("computing time = ", (time_stop - time_start)/1000.0, "[s]")
