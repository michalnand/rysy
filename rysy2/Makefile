LIBS_PATH=$(HOME)/libs/rysy/rysy2

export LIBS_PATH

all:
	cd libs_cnn && make -j4
	cd libs_common && make -j4

	cd tests && make -j4


clean:
	cd libs_cnn && make clean
	cd libs_common && make clean
	cd tests && make clean
