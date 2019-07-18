import sys
sys.path.append("../libs_rysy_python")


from rysy import *
import time



input_shape = Shape(80, 80, 24)
output_shape = Shape(1, 1, 8)

cnn = CNN(input_shape, output_shape, 0.0025)

cnn.add_layer("convolution", Shape(3, 3, 48))
cnn.add_layer("elu")
cnn.add_layer("max_pooling", Shape(2, 2))

cnn.add_layer("convolution", Shape(3, 3, 64))
cnn.add_layer("elu")
cnn.add_layer("max_pooling", Shape(2, 2))

cnn.add_layer("convolution", Shape(3, 3, 64))
cnn.add_layer("elu")
cnn.add_layer("max_pooling", Shape(2, 2))

cnn.add_layer("fc", Shape(1, 1, 256))
cnn.add_layer("elu")

cnn.add_layer("dropout")
cnn.add_layer("output")


'''
input_shape = Shape(19, 19, 4)
output_shape = Shape(1, 1, 362)

cnn = CNN(input_shape, output_shape, 0.0025)


cnn.add_layer("dense convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("dense convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("dense convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("dense convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("convolution", Shape(1, 1, 64))

cnn.add_layer("dense convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("dense convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("dense convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("dense convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("convolution", Shape(1, 1, 64))

cnn.add_layer("dense convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("dense convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("dense convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("dense convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("convolution", Shape(1, 1, 64))

cnn.add_layer("dropout")
cnn.add_layer("output")
'''



cnn._print()


input = Tensor(input_shape)
output = Tensor(output_shape)


iterations = 1000

time_start = int(round(time.time() * 1000))

for i in range(0, iterations):
    input.set_random(1.0)
    cnn.forward(output, input)

time_stop = int(round(time.time() * 1000))

print("forward time = ", (time_stop - time_start)/iterations, "[ms]")



time_start = int(round(time.time() * 1000))

for i in range(0, iterations):
    input.set_random(1.0)
    output.set_random(1.0)
    cnn.train(output, input)

time_stop = int(round(time.time() * 1000))

print("backward time = ", (time_stop - time_start)/iterations, "[ms]")
