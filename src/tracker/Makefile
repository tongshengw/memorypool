CC = gcc
all: test

test: main.o trackedmalloc.o
	$(CC) -o test main.o trackedmalloc.o
	
main.o: main.c trackedmalloc.h
	$(CC) -c main.c

trackedmalloc.o: trackedmalloc.c trackedmalloc.h
	$(CC) -c trackedmalloc.c