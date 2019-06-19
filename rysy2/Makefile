LIBS_PATH=$(HOME)/libs/rysy/rysy2

export LIBS_PATH

all:
	cd libs_nn && make -j4
	cd libs_common && make -j4
	cd libs_dataset && make -j4
	cd libs_experiment && make -j4

	cd libs_cnn_deployment && make -j4

	cd tests && make -j4


clean:
	cd libs_nn && make clean
	cd libs_common && make clean
	cd libs_dataset && make clean
	cd libs_experiment && make clean

	cd libs_cnn_deployment && make clean

	cd tests && make clean
