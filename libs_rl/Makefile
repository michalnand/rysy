OUTPUT_NAME = libs_rl

INC+= -I.

SRCS+= $(shell find . -name '*.cpp')

# CUDA_PATH = /usr/local/cuda-9.1/bin/

CC = g++
CU = nvcc
LD = g++
OBJCOPY = objcopy
OBJDUMP = objdump
SIZE = size
ARCHIVE = ar

# DEGUBFLAGS = -g -pg

CFLAGS = $(DEGUBFLAGS) -O2 -Os -fPIC -s -Wall -Wextra -pedantic -std=c++11 $(INC)

OBJS = $(SRCS:.cpp=.o)

all: $(OBJS)
	$(ARCHIVE) rc $(OUTPUT_NAME).a $(OBJS)
	$(CC) -shared -fPIC -o $(OUTPUT_NAME).so $(OBJS)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	rm $(OBJS) -rf
	rm $(OUTPUT_NAME).* -rf
