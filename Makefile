# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS = -g3 -Wall -I./linalg -I./phy
LDFLAGS =
CUDAFLAGS = -g3 -Xcompiler -Wall

# Sources
C_SOURCES = $(wildcard linalg/*.c) $(wildcard phy/*.c) main.c
CU_SOURCES = $(wildcard linalg/*.cu) $(wildcard phy/*.cu)
C_OBJS = $(C_SOURCES:.c=.o)
CU_OBJS = $(CU_SOURCES:.cu=.o)

# Target
TARGET = run

all: $(TARGET)

$(TARGET): $(C_OBJS) $(CU_OBJS)
	#$(NVCC) $(CUDAFLAGS) -o $@ $^
	$(CC) $(CUDAFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c -o $@ $<

clean:
	rm -f $(C_OBJS) $(CU_OBJS) $(TARGET)

