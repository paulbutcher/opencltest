opencltest: opencltest.c
	gcc -std=c99 -framework OpenCL opencltest.c -o opencltest
