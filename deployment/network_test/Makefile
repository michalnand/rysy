ELF = bin/main.elf

LIBS_PATH=$(HOME)/rysy


INC+= -I.
INC+= -I../libs_network
INC+= -I$(LIBS_PATH)/libs_dataset


LIBS+= ../libs_network/libs_network.a
LIBS+= $(LIBS_PATH)/libs_dataset/libs_dataset.a
LIBS+= $(LIBS_PATH)/libs_common/libs_common.a



CC = g++
CU = nvcc
LD = g++
OBJCOPY = objcopy
OBJDUMP = objdump
SIZE = size

# DEGUBFLAGS = -g -pg

CFLAGS = $(DEGUBFLAGS) -O2 -Os -s -Wall -Wextra -pedantic -std=c++11 $(INC)

LDFLAGS = $(DEGUBFLAGS) -lm -lpthread -pthread -std=c++11 -lglut -lGL -lGLU -lX11 -ljsoncpp -L/usr/local/cuda/lib64 -lcuda -lcudart


SRCS+= $(shell find -name '*.cpp')


OBJS = $(SRCS:.cpp=.o)

all: $(ELF)
	$(SIZE) $(ELF)

$(ELF): $(LIBS) $(OBJS)
	$(LD) -o $@ $(OBJS) $(LIBS) $(LDFLAGS)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@


clean:
	rm $(OBJS) -rf
	rm $(ELF) -rf
