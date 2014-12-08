
PROJ_BASE	:= $(HOME)/src/Folding

# https://github.com/isazi/utils
ISA_UTILS_INCLUDE	  := $(HOME)/src/utils/include
# https://github.com/isazi/AstroData
ISA_ASTRO_INCLUDE	  := $(HOME)/src/AstroData/include

CPU_INCLUDES	:= -I"$(PROJ_BASE)/include" -I"$(ISA_UTILS_INCLUDE)" -I"$(ISA_ASTRO_INCLUDE)"
PHI_INCLUDES	:= -I"$(PROJ_BASE)/include" -I"$(ISA_UTILS_INCLUDE)" -I"$(ISA_ASTRO_INCLUDE)"

ifneq ($(debug), 1)
	CFLAGS		  := -std=c++11 -O3 -g0 -Wall
	CPU_CFLAGS	:= -std=c++11 -O3 -g0 -Wall -mavx -fopenmp
	PHI_CFLAGS	:= -std=c++11 -O3 -g0 -Wall -openmp -vec-report3
else
	CFLAGS		  := -std=c++11 -O0 -g3 -Wall
	CPU_CFLAGS	:= -std=c++11 -O0 -g3 -Wall -mavx -fopenmp
	PHI_CFLAGS	:= -std=c++11 -O0 -g3 -Wall -openmp -vec-report3
endif

CPU_LDFLAGS		:= -lm
PHI_LDFLAGS		:= -lm

CC := icc

