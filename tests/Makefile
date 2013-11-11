
debug		:= 0
include		../Makefile.inc


all: clean Folding Transpose
 

Folding: Folding.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/FoldingTest Folding.cpp $(UTILS_SRC) $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

Transpose: Transpose.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/TransposeTest Transpose.cpp $(UTILS_SRC) $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/FoldingTest
	rm -f $(PROJ_BASE)/bin/TransposeTest

