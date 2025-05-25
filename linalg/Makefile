# Compiler and flags
CC = gcc
CFLAGS = -g3 -Wall
OBJ = leastsq.o lubksb.o ludcmp.o luminv.o main.o mvdot.o vvdot.o

# Default target
all: run 

# Linking
run: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ)

# Compile .c to .o
%.o: %.c linalg.h
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f *.o main

.PHONY: all clean

