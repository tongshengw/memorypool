# Compiler and flags
CC = gcc
CFLAGS = -g3 -Wall
OBJ = leastsq_kkt.o leastsq.o lubksb.o ludcmp.o luminv.o main.o mvdot.o vvdot.o mmdot.o
POOLOBJ = leastsq_kkt.o leastsq.o lubksb.o ludcmp.o luminv.o pool-main.o mvdot.o vvdot.o mmdot.o poolalloc.o

# Default target
all: run 
	
pool: pool-run

pool-run: $(POOLOBJ)
	$(CC) $(CFLAGS) -o $@ $(POOLOBJ)

# Linking
run: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ)

# Compile .c to .o
%.o: %.c linalg.h
	$(CC) $(CFLAGS) -c $< -o $@
	
poolalloc.o: ../cpu-concept/poolalloc.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f *.o run

.PHONY: all clean

