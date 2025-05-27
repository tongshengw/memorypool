# Compiler settings
CC = gcc
NVCC = nvcc
CURRENT_DIR = $(shell pwd)
CFLAGS = -g3 -Wall -I$(CURRENT_DIR)
LDFLAGS =
CUDAFLAGS = -g3 -Xcompiler -Wall

# Sources
HEADERS = $(wildcard linalg/*.h) $(wildcard phy/*.h)
C_SOURCES = $(wildcard linalg/*.c) $(wildcard phy/*.c)
CU_SOURCES = $(wildcard linalg/*.cu) $(wildcard phy/*.cu)
C_OBJS = $(C_SOURCES:.c=.o)
CU_OBJS = $(CU_SOURCES:.cu=.o)

# Target
TARGET = run

all: $(TARGET)

$(TARGET): $(C_OBJS) $(CU_OBJS)
	#$(NVCC) $(CUDAFLAGS) -o $@ $^
	$(CC) $(CFLAGS) main.c -o $@ $^

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu $(HEADERS)
	$(NVCC) $(CUDAFLAGS) -c -o $@ $<

clean:
	rm -f $(C_OBJS) $(CU_OBJS) $(TARGET)
