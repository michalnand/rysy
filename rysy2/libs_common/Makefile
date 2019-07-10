OUTPUT_NAME = libs_common

INC+= -I.
INC+= -I../libs_nn
INC+= -I../libs_dataset
INC+= -I../libs_common

SRCS+= $(shell find . -name '*.cpp')
SRCS_CUDA+= $(shell find . -name '*.cu')

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
CUDAFLAGS = -Xptxas -O3 -arch=sm_50 -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC -std=c++11 $(INC)

OBJS = $(SRCS:.cpp=.o)
OBJS_CUDA = $(SRCS_CUDA:.cu=.oc)

all: $(OBJS) $(OBJS_CUDA)
	$(ARCHIVE) rc $(OUTPUT_NAME).a $(OBJS) $(OBJS_CUDA)
	$(CC) -shared -fPIC -o $(OUTPUT_NAME).so $(OBJS) $(OBJS_CUDA)

%.o: %.cpp
	$(CC) -c $(CFLAGS) $< -o $@

%.oc: %.cu
	$(CUDA_PATH)$(CU) -c $(CUDAFLAGS) $< -o $@

clean:
	rm $(OBJS) -rf
	rm $(OBJS_CUDA) -rf
	rm $(OUTPUT_NAME).* -rf
