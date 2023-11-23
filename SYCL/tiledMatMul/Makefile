include ../Makefile.in

ifeq ($(TARGET), cuda)
	SRC = $(shell find . -type f -name *.cu)
	EXE = $(subst .cu,,$(SRC))_cuda
else 
	SRC = $(shell find . -type f -name *.cpp)
	_EXE = $(subst .cpp,,$(SRC))
	
	ifeq ($(BACKEND), cpu)
		EXE = $(_EXE)_sycl_cpu
	else
		EXE = $(_EXE)_sycl_gpu
	endif

endif

$(EXE): $(SRC)
	@echo -n "building $@ ..."
	@$(CXX) $< -o $@ $(CXX_FLAGS) -I ../utils
	@echo " done"

.PHONY: clean
clean:
	rm -f $(subst .cpp,,$(SRC))_*

