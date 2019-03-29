ELF = bin/deployment_test.elf

#LIBS_PATH=$(HOME)/libs/rysy

INC+= -I.
INC+= -I$(LIBS_PATH)/libs_common
INC+= -I$(LIBS_PATH)/libs_cnn_deployment

LIBS+= $(LIBS_PATH)/libs_cnn_deployment/libs_cnn_deployment.a
LIBS+= $(LIBS_PATH)/libs_common/libs_common.a

CC = g++
CU = nvcc
LD = g++
OBJCOPY = objcopy
OBJDUMP = objdump
SIZE = size

# DEGUBFLAGS = -g -pg

CFLAGS = $(DEGUBFLAGS) -O2 -Os -s -Wall -Wextra -pedantic -std=c++11 $(INC)
CUDAFLAGS = -O3 -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC -std=c++11 $(INC)

LDFLAGS = $(DEGUBFLAGS) -lstdc++fs -lm -lpthread -pthread -std=c++11 -lglut -lGL -lGLU -lX11 `pkg-config --libs opencv` -ljsoncpp -L/usr/local/cuda/lib64 -lcuda -lcudart


SRCS+= $(shell find -name '*.cpp')
SRCS_CUDA+= $(shell find -name '*.cu')


OBJS = $(SRCS:.cpp=.o)
OBJS_CUDA = $(SRCS_CUDA:.cu=.oc)

all: $(ELF)
	$(SIZE) $(ELF)

$(ELF): $(LIBS) $(OBJS) $(OBJS_CUDA)
	$(LD) -o $@ $(OBJS) $(OBJS_CUDA) $(LIBS) $(LDFLAGS)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

%.oc: %.cu
	$(CU) -c $(CUDAFLAGS) $< -o $@

clean:
	rm $(OBJS) -rf
	rm $(OBJS_CUDA) -rf
	rm $(ELF) -rf
