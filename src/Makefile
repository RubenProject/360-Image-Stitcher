CXX ?= g++

CXXFLAGS += -c -Wall -std=c++11 $(shell pkg-config --cflags opencv)
LDFLAGS += $(shell pkg-config --libs --static opencv)
OBJS = astar.o projection.o imgops.o stitcher.o manual.o main.o

main: $(OBJS)
	$(CXX) $(OBJS) -o stitch $(LDFLAGS)

manual.o: manual.cpp manual.hpp orientation.hpp
	$(CXX) $(CXXFLAGS) manual.cpp

imgops.o: imgops.cpp imgops.hpp
	$(CXX) $(CXXFLAGS) imgops.cpp

projection.o: projection.cpp projection.hpp
	$(CXX) $(CXXFLAGS) projection.cpp

astar.o: astar.cpp astar.hpp
	$(CXX) $(CXXFLAGS) astar.cpp

stitcher.o: stitcher.cpp stitcher.hpp astar.hpp projection.hpp imgops.hpp manual.hpp orientation.hpp
	$(CXX) $(CXXFLAGS) stitcher.cpp

main.o: main.cpp stitcher.hpp projection.hpp imgops.hpp manual.hpp orientation.hpp
	$(CXX) $(CXXFLAGS) main.cpp 

clean: ; rm -f stitcher.o astar.o projection.o imgops.o manual.o main.o astar.o program
