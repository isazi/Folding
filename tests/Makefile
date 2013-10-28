
debug		:= 0
include		../Makefile.inc


all: clean Folding
 

Folding: Folding.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/FoldingTest Folding.cpp $(UTILS_SRC) $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

clean:
	rm -f $(PROJ_BASE)/bin/FoldingTest

