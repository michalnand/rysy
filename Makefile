
all:
	cd libs_common && make -j4
	cd libs_dataset && make -j4
	cd libs_nn && make -j4
	cd libs_experiment && make -j4
	cd libs_rl && make -j4
	cd deployment && make -j4
	cd examples/classification_test && make -j4
	cd examples/rl && make -j4
 


clean:
	cd libs_common && make clean
	cd libs_dataset && make clean
	cd libs_nn && make clean
	cd libs_experiment && make clean
	cd libs_rl && make clean
	cd deployment && make clean
	cd examples/classification_test && make clean
	cd examples/rl && make clean
