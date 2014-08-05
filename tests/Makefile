
debug		:= 0
include		../Makefile.inc


all: clean Folding
 

Folding: Folding.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/FoldingTest Folding.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS)

FoldingAVX: FoldingAVX.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/FoldingAVXTest FoldingAVX.cpp $(AVX_INCLUDES) $(AVX_LIBS) $(AVX_CFLAGS) $(AVX_LDFLAGS)

FoldingPhi: FoldingPhi.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/FoldingPhiTest FoldingPhi.cpp $(PHI_INCLUDES) $(PHI_LIBS) $(PHI_CFLAGS) $(PHI_LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/FoldingTest
	rm -f $(PROJ_BASE)/bin/FoldingAVXTest
	rm -f $(PROJ_BASE)/bin/FoldingPhiTest
