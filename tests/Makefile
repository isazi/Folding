
debug		:= 0
include		../Makefile.inc


all: clean Folding 
 

Folding: Folding.cpp
	$(CC) -o $(PROJ_BASE)/bin/FoldingTest Folding.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/FoldingTest
