CXX ?= g++

CXXFLAGS += -c -Wall -std=c++11 $(shell pkg-config --cflags opencv)
LDFLAGS += $(shell pkg-config --libs --static opencv)
OBJS = AStar.o stitcher.o

stitcher: $(OBJS)
	$(CXX) $(OBJS) -o stitcher $(LDFLAGS)

AStar.o: AStar.cpp AStar.hpp
	$(CXX) $(CXXFLAGS) AStar.cpp

stitcher.o: stitcher.cpp AStar.hpp
	$(CXX) stitcher.cpp $(CXXFLAGS)

clean: ; rm -f stitcher.o astar.o stitcher
