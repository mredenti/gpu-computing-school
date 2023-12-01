include ../Makefile.in

ifeq ($(TARGET), cuda)
	FILEXT = .cu
	SUFFIX = $(TARGET)
else 
	FILEXT = .cpp
	SUFFIX = $(TARGET)_$(BACKEND)
endif

# targets
SRC = $(shell find . -type f -name *$(FILEXT))
EXE = $(subst $(FILEXT),,$(SRC))_$(SUFFIX)
	
$(EXE): $(SRC)
	@echo -n "building $@ ..."
	@$(CXX) $< -o $@ $(CXX_FLAGS) -I ../utils
	@echo " done"

.PHONY: clean
clean:
	rm -f $(subst .cpp,,$(SRC))_*
