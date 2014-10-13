
debug		:= 0
include		../Makefile.inc


all: clean Folding FoldingAVX FoldingPhi FoldingDEBUG
 

Folding: Folding.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/FoldingTest Folding.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

FoldingAVX: FoldingSIMD.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/FoldingAVXTest FoldingSIMD.cpp $(CPU_INCLUDES) $(CPU_LIBS) $(CPU_CFLAGS) $(CPU_LDFLAGS)

FoldingPhi: FoldingSIMD.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/FoldingPhiTest FoldingSIMD.cpp $(PHI_INCLUDES) $(PHI_LIBS) $(PHI_CFLAGS) $(PHI_LDFLAGS)

FoldingDEBUG: FoldingDEBUG.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/FoldingDEBUG FoldingDEBUG.cpp $(CPU_INCLUDES) $(CPU_LIBS) $(CPU_CFLAGS) $(CPU_LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/FoldingTest
	rm -f $(PROJ_BASE)/bin/FoldingAVXTest
	rm -f $(PROJ_BASE)/bin/FoldingPhiTest
	rm -f $(PROJ_BASE)/bin/FoldingDEBUG
