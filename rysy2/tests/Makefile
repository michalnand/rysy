INC+= -I.
INC+= -I$(LIBS_PATH)/libs_nn
INC+= -I$(LIBS_PATH)/libs_dataset
INC+= -I$(LIBS_PATH)/libs_common

LIBS+= $(LIBS_PATH)/libs_dataset/libs_dataset.a
LIBS+= $(LIBS_PATH)/libs_nn/libs_nn.a
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

LDFLAGS = $(DEGUBFLAGS) -lstdc++fs -lm -lpthread -pthread -std=c++11 -lglut -lGL -lGLU -lX11 -ljsoncpp -L/usr/local/cuda/lib64 -lcuda -lcudart

SRCS+= $(shell find -name '*.cpp')
OBJS = $(SRCS:.cpp=.o)

ELFS = $(SRCS:.cpp=.elf)

all: $(ELFS)

%.elf: %.o
	$(LD) -o $@ $< $(LIBS) $(LDFLAGS)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm $(OBJS) -f
	rm $(ELFS) -f
