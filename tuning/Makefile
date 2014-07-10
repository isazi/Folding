
debug		:= 0
include		../Makefile.inc


all: clean Folding

Folding: Folding.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/FoldingTuning Folding.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/FoldingTuning
