CXX = g++

# CXXFLAGS = -std=c++11 -O3
ARCH_FLAGS = -march=ivybridge -mtune=ivybridge
# ARCH_FLAGS = -march=native -mtune=native
FP_FLAGS = -ffast-math -mfpmath=sse
VEC_FLAGS = -msse -msse2 -msse3 -msse4.1 -ftree-vectorize -fopt-info-vec
CXXFLAGS = -std=c++11 -O3 $(ARCH_FLAGS) $(FP_FLAGS) $(VEC_FLAGS) -funroll-loops

LDFLAGS = -lm -flto

BINARIES = test strassen strassen_partial strassen_recursion_size

COMMON_SOURCES = utils.cpp

STRASSEN_SOURCES = main.cpp strassen.cpp $(COMMON_SOURCES)

STRASSEN_PARTIAL_SOURCES = main.cpp strassen_partial.cpp $(COMMON_SOURCES)

STRASSEN_RECURSION_SIZE_SOURCES = main_recursion_size.cpp strassen_partial.cpp $(COMMON_SOURCES)

TEST_SOURCES = test_main.cpp strassen_partial.cpp simple.cpp $(COMMON_SOURCES)

HEADERS = strassen.h utils.h

.PHONY: all
all: $(BINARIES)

strassen: $(STRASSEN_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(STRASSEN_SOURCES)

strassen_partial: $(STRASSEN_PARTIAL_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(STRASSEN_PARTIAL_SOURCES)

strassen_recursion_size: $(STRASSEN_RECURSION_SIZE_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(STRASSEN_RECURSION_SIZE_SOURCES)

test: $(TEST_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(TEST_SOURCES)

.PHONY: clean
clean:
	rm $(BINARIES)
