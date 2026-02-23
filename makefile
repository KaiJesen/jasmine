
TARGET = matrix_test
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -pedantic -g
SRCS = main.cpp 
OBJS = $(SRCS:.cpp=.o)


$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

all: $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean