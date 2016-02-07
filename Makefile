CC=g++
CFLAGS=-ansi -pedantic -Wno-deprecated
INC=-I$(EIGEN3_INCLUDE_DIR) -I./
LIB=-DGL_GLEXT_PROTOTYPES -lglut -lGL -lGLU

all:
	$(CC) $(CFLAGS) $(INC) *.cpp $(LIB) -o a1

run:
	./a1
clean:
	rm -f *~ *.o a.out a1
clear:
	clear
	rm -f *~ *.o a.out a1
	$(CC) $(CFLAGS) $(INC) *.cpp $(LIB) -o a1
cc:
	rm -f *~ *.o a.out a1
	clear
	$(CC) $(CFLAGS) $(INC) *.cpp $(LIB) -o a1
fast:
	rm -f *~ *.o a.out a1
	clear
	$(CC) $(CFLAGS) $(INC) *.cpp $(LIB) -o a1
	./a1