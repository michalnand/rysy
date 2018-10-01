
all:
	cd libs_common && make -j4
	cd libs_dataset && make -j4
	cd libs_nn && make -j4
	cd libs_experiment && make -j4
	cd deployment && make -j4
	cd examples/classification_test && make -j4
	cd examples/autoencoder_test && make -j4
	cd examples/speed_test && make -j4
	cd examples/convolution_speed_test && make -j4
	cd examples/classification_compare && make -j4



clean:
	cd libs_common && make clean
	cd libs_dataset && make clean
	cd libs_nn && make clean
	cd libs_experiment && make clean
	cd deployment && make clean
	cd examples/classification_test && make clean
	cd examples/autoencoder_test && make clean
	cd examples/speed_test && make clean
	cd examples/classification_compare && make clean
