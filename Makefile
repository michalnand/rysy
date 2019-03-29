LIBS_PATH=$(HOME)/libs/rysy

export LIBS_PATH

all:
	cd libs_common && make -j4
	cd libs_dataset && make -j4
	cd libs_nn && make -j4
	cd libs_preprocessing && make -j4
	cd libs_experiment && make -j4

	cd libs_cnn_deployment && make -j4

	cd libs_rysy_python && make -j4

	cd deployment && make -j4
	cd examples/autoencoder_test && make -j4
	cd examples/classification_compare && make -j4
	cd examples/convolution_speed_test && make -j4
	cd examples/convolution_test && make -j4
	cd examples/images_classification_test && make -j4
	cd examples/preprocessing_test && make -j4
	cd examples/regression_test && make -j4
	cd examples/denoising_autoencoder && make -j4
	cd examples/deployment_test && make -j4


clean:
	cd libs_common && make clean
	cd libs_dataset && make clean
	cd libs_nn && make clean
	cd libs_preprocessing && make clean
	cd libs_experiment && make clean

	cd libs_cnn_deployment && make clean

	cd libs_rysy_python && make clean

	cd deployment && make clean
	cd examples/autoencoder_test && make clean
	cd examples/classification_compare && make clean
	cd examples/convolution_speed_test && make clean
	cd examples/convolution_test && make clean
	cd examples/images_classification_test && make clean
	cd examples/preprocessing_test && make clean
	cd examples/regression_test && make clean
	cd examples/denoising_autoencoder && make clean

	cd examples/deployment_test && make clean
