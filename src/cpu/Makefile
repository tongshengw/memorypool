CC = gcc
CFLAGS = -Wall -g3 -std=c99 -Werror
TARGETS = demo fuzzy

fuzzy : fuzzy-test.o poolalloc.o
	$(CC) -o fuzzy fuzzy-test.o poolalloc.o $(CFLAGS)

demo : demo.o poolalloc.o
	$(CC) -o demo demo.o poolalloc.o $(CFLAGS)

fuzzy-test.o : fuzzy-test.c
	$(CC) -c fuzzy-test.c poolalloc.c $(CFLAGS)

demo.o : demo.c
	$(CC) -c demo.c poolalloc.c $(CFLAGS)

poolalloc.o : poolalloc.c
	$(CC) -c poolalloc.c $(CFLAGS)

clean:
	rm -f *.o $(TARGETS)
