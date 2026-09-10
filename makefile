
TARGET = matrix_test
CXX = g++
CXXFLAGS = -std=c++20 -mavx2 -Wall -Wextra -pedantic -g -fopenmp -O3
# -lprofiler must come after .o and with --no-as-needed, or the linker drops it (no prof.data).
LDFLAGS = -fopenmp -Wl,--no-as-needed -lprofiler -Wl,--as-needed
SRCS = main.cpp 
OBJS = $(SRCS:.cpp=.o)


$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

all: $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean