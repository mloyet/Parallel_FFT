TARGET = fft
OBJS = fft.o
OBJS += main.o
OBJS += fft_ispc.o

CC = gcc
CFLAGS = -std=gnu99 -Wall -Werror -g -O3 -fopenmp
LDFLAGS = -lfftw3 -lm -lpthread

ISPC=ispc
ISPCFLAGS=-O2 --target=avx1-i32x8 --arch=x86-64 --math-lib=default

default: $(TARGET)
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

fft_ispc.o: fft.ispc
	$(ISPC) $(ISPCFLAGS) fft.ispc -o fft_ispc.o -h fft_ispc.h

DEPS = $(OBJS:%.o=%.d)
-include $(DEPS)

clean:
	rm $(TARGET) $(OBJS) $(DEPS) || true