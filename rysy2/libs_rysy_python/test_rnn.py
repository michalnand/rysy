from rysy import *

#load dataset
dataset_path = "/home/michal/dataset/mnist/"
dataset = DatasetTest(1)


input_shape = dataset.get_input_shape()
output_shape = dataset.get_output_shape()


cnn = RNN(input_shape, output_shape, 0.001)


cnn.add_layer("recurrent", Shape(1, 1, 128))

'''
cnn.add_layer("fc", Shape(1, 1, 64))
cnn.add_layer("elu")
cnn.add_layer("fc", Shape(1, 1, 64))
cnn.add_layer("elu")
'''
cnn.add_layer("output")

cnn._print()





#train network - set epoch count

for epoch in range(0, 1):
    print("\n\n\nEPOCH = ", epoch, "\n")
    cnn.train(dataset.get_training_output_all(), dataset.get_training_input_all(), 1)

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
    print(compare.asString())

print("program done")
