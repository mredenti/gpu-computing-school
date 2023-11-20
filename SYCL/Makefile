CXX = syclcc
CXX_FLAGS = -O3

# select backend
CXX_FLAGS += --opensycl-targets=cuda-nvcxx # nvcxx backend
# CXX_FLAGS += -fopenmp --opensycl-targets='cuda:sm_70' # clang + cuda backend

SRC = $(wildcard *.dp.cpp)
EXE = $(SRC:.dp.cpp=.x)

default: $(EXE)

%.x: %.dp.cpp
	$(CXX) $< -o $@ $(CXX_FLAGS) 

clean:
	rm -rf $(EXE)
