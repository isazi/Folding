
debug		:= 0
include		../Makefile.inc


all: clean Folding Transpose
 

Folding: Folding.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/FoldingTuning Folding.cpp $(UTILS_SRC) $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

Transpose: Transpose.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/TransposeTuning Transpose.cpp $(UTILS_SRC) $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/FoldingTuning
	rm -f $(PROJ_BASE)/bin/TransposeTuning
