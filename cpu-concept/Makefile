CC = g++
CFLAGS = -Wall -g3 -std=c++17 -Werror
TARGET = demo

demo : demo.o poolalloc.o
	$(CC) -o demo demo.o poolalloc.o $(CFLAGS)

demo.o : demo.cpp
	$(CC) -c demo.cpp poolalloc.cpp $(CFLAGS)

poolalloc.o : poolalloc.cpp
	$(CC) -c poolalloc.cpp $(CFLAGS)

clean:
	rm -f *.o $(TARGET)
