import sys
sys.path.append("../libs_rysy_python")


from rysy import *


#dataset = DatasetTest(1)


dataset_path = "/home/michal/dataset/mnist/"
dataset = DatasetMnistRNN(dataset_path + "train-images-idx3-ubyte",
                            dataset_path + "train-labels-idx1-ubyte",
                            dataset_path + "t10k-images-idx3-ubyte",
                            dataset_path + "t10k-labels-idx1-ubyte")


input_shape = dataset.get_input_shape()
output_shape = dataset.get_output_shape()


cnn = RNN(input_shape, output_shape, 0.001)


#cnn.add_layer("fc", Shape(16))
cnn.add_layer("fir")
cnn.add_layer("fc", Shape(256))
cnn.add_layer("elu")
cnn.add_layer("output")

cnn._print()

#train network - set epoch count

rnn_log = Log("rnn.log")
for epoch in range(0, 10):
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

    log_str = "epoch = " + str(epoch) + "\n\n"
    log_str+= str(compare.asString())
    log_str+= "\n\n\n\n"
    rnn_log.put_string(log_str)



print("program done")
