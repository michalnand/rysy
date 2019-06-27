OUTPUT_NAME = rysy

INC+= -I.
INC+= -I$(LIBS_PATH)/libs_nn
INC+= -I$(LIBS_PATH)/libs_dataset
INC+= -I$(LIBS_PATH)/libs_common
INC+= -I$(LIBS_PATH)/libs_experiment
INC+= -I$(LIBS_PATH)/libs_dqn
INC+= -I$(LIBS_PATH)/libs_embedded_export
INC+= -I$(LIBS_PATH)/libs_embedded_network

SRCS+= $(shell find . -name '*.cpp')
SRCS_SWIG+= $(shell find . -name '*.i')

CC = g++
CU = nvcc
LD = g++
SWIG = swig3.0
OBJCOPY = objcopy
OBJDUMP = objdump
SIZE = size
ARCHIVE = ar


PYTHONI = $(shell python3-config --cflags)
PYTHONL = $(shell python3-config --ldflags)

CFLAGS+= -O2 -Os -fPIC -s -Wall -Wextra -pedantic -std=c++17 $(INC)

LD_OPENGLFLAGS = -lglut -lGL -lGLU -lX11
LD_CUDAFLAGS 	 = -L/usr/local/cuda/lib64 -lcuda -lcudart
LD_COMMONFLAGS = -ljsoncpp

LDFLAGS = $(LD_OPENGLFLAGS) $(LD_COMMONFLAGS) $(LD_CUDAFLAGS) -lm -lpthread -pthread -std=c++17 -lstdc++fs


OBJS 				= $(SRCS:.cpp=.o)
OBJS_CUDA 	= $(SRCS_CUDA:.cu=.oc)
OBJS_CSWIG  = $(SRCS_SWIG:.i=.cxx)
OBJS_SWIG 	= $(OBJS_CSWIG:.cxx=.oswig)


OBJS_EXTERN+= $(shell find ~/libs/rysy/rysy2/libs_experiment -name '*.o')
OBJS_EXTERN+= $(shell find ~/libs/rysy/rysy2/libs_dataset -name '*.o')
OBJS_EXTERN+= $(shell find ~/libs/rysy/rysy2/libs_nn -name '*.o')
OBJS_EXTERN+= $(shell find ~/libs/rysy/rysy2/libs_nn -name '*.oc')
OBJS_EXTERN+= $(shell find ~/libs/rysy/rysy2/libs_common -name '*.o')
OBJS_EXTERN+= $(shell find ~/libs/rysy/rysy2/libs_dqn -name '*.o')
OBJS_EXTERN+= $(shell find ~/libs/rysy/rysy2/libs_embedded_export -name '*.o')
OBJS_EXTERN+= $(shell find ~/libs/rysy/rysy2/libs_embedded_network -name '*.o')


all: $(OBJS) $(OBJS_CUDA) $(OBJS_SWIG)
	 $(CC) -shared -fPIC $(PYTHONL) -o _$(OUTPUT_NAME).so $(OBJS) $(OBJS_EXTERN) $(OBJS_SWIG) $(LDFLAGS)



%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

%.oc: %.cu
	$(CUDA_PATH)$(CU) -c $(CUDAFLAGS) $< -o $@


%.cxx: %.i
	$(SWIG) -c++ -python $(INC) -o $@ $<

%.oswig: %.cxx
	$(CC) -c -fPIC $(INC) $(PYTHONI) $< -o $@

clean:
	rm $(OBJS) -f
	rm $(OBJS_CSWIG) -f
	rm $(OBJS_SWIG) -f
	rm _$(OUTPUT_NAME).so -f
	rm $(OUTPUT_NAME).py -f
	rm $(OUTPUT_NAME).pyc -f
