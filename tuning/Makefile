
debug		:= 0
include		../Makefile.inc


all: clean Folding FoldingAVX FoldingPhi
 

Folding: Folding.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/FoldingTuning Folding.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

FoldingAVX: FoldingSIMD.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/FoldingAVXTuning FoldingSIMD.cpp $(CPU_INCLUDES) $(CPU_CFLAGS) $(CPU_LDFLAGS)

FoldingPhi: FoldingSIMD.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/FoldingPhiTuning FoldingSIMD.cpp $(PHI_INCLUDES) $(PHI_CFLAGS) $(PHI_LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/FoldingTuning
	rm -f $(PROJ_BASE)/bin/FoldingAVXTuning
	rm -f $(PROJ_BASE)/bin/FoldingPhiTuning
