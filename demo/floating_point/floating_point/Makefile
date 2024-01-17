all: loop_int loop_float loop_float_revisited precision precision_2 precision_3 count

#EXTRA_CFLAGS = -DHAVE_LINUX_PERF_EVENT_H 
EXTRA_CFLAGS = -O3 -g

loop_int: loop_int.c 
	gcc $(EXTRA_CFLAGS) loop_int.c -o loop_int

loop_float: loop_float.c 
	gcc $(EXTRA_CFLAGS) loop_float.c -o loop_float

loop_float_revisited: loop_float_revisited.c 
	gcc $(EXTRA_CFLAGS) loop_float_revisited.c -o loop_float_revisited

precision: precision.c 
	gcc $(EXTRA_CFLAGS) precision.c -o precision

precision_2: precision_2.c 
	gcc $(EXTRA_CFLAGS) precision_2.c -o precision_2

precision_3: precision_3.c 
	gcc $(EXTRA_CFLAGS) precision_3.c -o precision_3

count: count.c 
	gcc -O0 count.c -o count

clean:
		rm -f loop_int loop_float loop_float_revisited precision precision_2 precision_3 count
