# Compiler and flags
CC = gcc
CFLAGS = -O3 -Wall
OBJ = leastsq_kkt.o leastsq.o lubksb.o ludcmp.o luminv.o main.o mvdot.o vvdot.o mmdot.o poolalloc.o
POOLOBJ = leastsq_kkt.o leastsq.o lubksb.o ludcmp.o luminv.o pool-main.o mvdot.o vvdot.o mmdot.o poolalloc.o
POOLVOLTESTOBJ = leastsq_kkt.o leastsq.o lubksb.o ludcmp.o luminv.o volume-test-pool.o mvdot.o vvdot.o mmdot.o poolalloc.o
VOLTESTOBJ = leastsq_kkt.o leastsq.o lubksb.o ludcmp.o luminv.o volume-test.o mvdot.o vvdot.o mmdot.o poolalloc.o

# Default target
all: run pool-run volume-test volume-test-pool

volume: volume-test-pool volume-test
	
volume-test-pool: $(POOLVOLTESTOBJ)
	$(CC) $(CFLAGS) -o $@ $(POOLVOLTESTOBJ)

volume-test: $(VOLTESTOBJ)
	$(CC) $(CFLAGS) -o $@ $(VOLTESTOBJ)

pool-run: $(POOLOBJ)
	$(CC) $(CFLAGS) -o $@ $(POOLOBJ)

# Linking
run: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ)

# Compile .c to .o
%.o: %.c linalg.h volume-test.h
	$(CC) $(CFLAGS) -c $< -o $@
	
poolalloc.o: ../cpu-concept/poolalloc.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f *.o run volume-test volume-test-pool pool-run

.PHONY: all clean

