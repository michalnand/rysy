from rysy import *
import time
import random

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

cnn = CNN(dataset.get_input_shape(), dataset.get_output_shape(), 0.0025)

cnn.add_layer("convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("max_pooling", Shape(2, 2))
cnn.add_layer("convolution", Shape(3, 3, 32))
cnn.add_layer("elu")
cnn.add_layer("max_pooling", Shape(2, 2))
cnn.add_layer("convolution", Shape(3, 3, 32))
cnn.add_layer("elu")
cnn.add_layer("dropout")
cnn.add_layer("output")

cnn._print()


#train network - set epoch count
epoch_count = 1



cnn.train(dataset.get_training_output_all(), dataset.get_training_input_all(), epoch_count)
#cnn.save("example_network/")


#test network response on whole testing dataset items

nn_output = VectorFloat(dataset.get_classes_count())

nn_visualisation = CNNVisualisation(cnn)


idx = random.randint(0, dataset.get_testing_count()-1)
cnn.forward(nn_output, dataset.get_testing_input(idx))
nn_visualisation.process()
#nn_visualisation.save("layers_images/")

while True:
    time.sleep(5)

print("\n\nprogram done\n")

while True:

    pass
