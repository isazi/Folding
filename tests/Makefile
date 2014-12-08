
debug		:= 0
include		../Makefile.inc


all: clean FoldingAVX FoldingPhi
 

FoldingAVX: FoldingSIMD.cpp
	$(CC) -o $(PROJ_BASE)/bin/FoldingAVXTest FoldingSIMD.cpp $(CPU_INCLUDES) $(CPU_LIBS) $(CPU_CFLAGS) $(CPU_LDFLAGS)

FoldingPhi: FoldingSIMD.cpp
	$(CC) -o $(PROJ_BASE)/bin/FoldingPhiTest FoldingSIMD.cpp $(PHI_INCLUDES) $(PHI_LIBS) $(PHI_CFLAGS) $(PHI_LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/FoldingAVXTest
	rm -f $(PROJ_BASE)/bin/FoldingPhiTest
