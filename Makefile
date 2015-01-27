
# https://github.com/isazi/utils
UTILS := $(HOME)/src/utils
# https://github.com/isazi/OpenCL
OPENCL := $(HOME)/src/OpenCL
# https://github.com/isazi/AstroData
ASTRODATA := $(HOME)/src/AstroData

INCLUDES := -I"include" -I"$(ASTRODATA)/include" -I"$(UTILS)/include"
CL_INCLUDES := $(INCLUDES) -I"$(OPENCL)/include"
CL_LIBS := -L"$(OPENCL_LIB)"

CFLAGS := -std=c++11 -Wall
ifneq ($(debug), 1)
	CFLAGS += -O3 -g0
else
	CFLAGS += -O0 -g3
endif

LDFLAGS := -lm
CL_LDFLAGS := $(LDFLAGS) -lOpenCL

CC := g++

# Dependencies
DEPS := $(ASTRODATA)/bin/Observation.o $(UTILS)/bin/ArgumentList.o $(UTILS)/bin/Timer.o $(UTILS)/bin/utils.o bin/Bins.o bin/Folding.o
CL_DEPS := $(DEPS) $(OPENCL)/bin/Exceptions.o $(OPENCL)/bin/InitializeOpenCL.o $(OPENCL)/bin/Kernel.o 


all: bin/Bins.o bin/Folding.o bin/FoldingTest bin/FoldingTuning bin/printCode

bin/Bins.o: $(ASTRODATA)/bin/Observation.o $(UTILS)/bin/utils.o include/Bins.hpp src/Bins.cpp
	$(CC) -o bin/Bins.o -c src/Bins.cpp $(INCLUDES) $(CFLAGS)

bin/Folding.o: $(UTILS)/bin/utils.o bin/Bins.o include/Folding.hpp src/Folding.cpp
	$(CC) -o bin/Folding.o -c src/Folding.cpp $(CL_INCLUDES) $(CFLAGS)

bin/FoldingTest: $(CL_DEPS) src/FoldingTest.cpp
	$(CC) -o bin/FoldingTest src/FoldingTest.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/FoldingTuning: $(CL_DEPS) src/FoldingTuning.cpp
	$(CC) -o bin/FoldingTuning src/FoldingTuning.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/printCode: $(DEPS) src/printCode.cpp
	$(CC) -o bin/printCode src/printCode.cpp $(DEPS) $(INCLUDES) $(LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*

