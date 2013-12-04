
debug		:= 0
include		../Makefile.inc


all: clean Folding FoldingAVX

Folding: Folding.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/FoldingTest Folding.cpp $(UTILS_SRC) $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

FoldingAVX: FoldingAVX.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/FoldingAVXTest FoldingAVX.cpp $(UTILS_SRC) $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/FoldingTest
	rm -f $(PROJ_BASE)/bin/FoldingAVXTest

