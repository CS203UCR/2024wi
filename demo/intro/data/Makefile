all: gen

EXTRA_CFLAGS = -O3 -mavx  -g

gen: gen.c 
	gcc  $(EXTRA_CFLAGS) gen.c -o gen $(LD_FLAGS)

clean:
		rm -f gen *~ *.o
