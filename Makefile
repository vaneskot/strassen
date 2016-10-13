CXX = g++

CXXFLAGS = -std=c++11 -O3

LDFLAGS = -lm

BINARIES = test strassen strassen_fully_partial

STRASSEN_SOURCES = main.cpp strassen.cpp

STRASSEN_FULLY_PARTIAL_SOURCES = main.cpp strassen_fully_partial.cpp

TEST_SOURCES = test_main.cpp strassen.cpp simple.cpp

HEADERS = strassen.h

.PHONY: all
all: $(BINARIES)

strassen: $(STRASSEN_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(STRASSEN_SOURCES)

strassen_fully_partial: $(STRASSEN_FULLY_PARTIAL_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(STRASSEN_FULLY_PARTIAL_SOURCES)

test: $(TEST_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(TEST_SOURCES)

.PHONY: clean
clean:
	rm $(BINARIES)
