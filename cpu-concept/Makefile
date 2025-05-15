CC = gcc
CFLAGS = -Wall -g3 -std=c99 -Werror
TARGET = demo

demo : demo.o poolalloc.o
	$(CC) -o demo demo.o poolalloc.o $(CFLAGS)

demo.o : demo.c
	$(CC) -c demo.c poolalloc.c $(CFLAGS)

poolalloc.o : poolalloc.c
	$(CC) -c poolalloc.c $(CFLAGS)

clean:
	rm -f *.o $(TARGET)
