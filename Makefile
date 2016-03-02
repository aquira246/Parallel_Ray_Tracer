CU=nvcc
CC=icpc
CFLAGS=-ansi -pedantic -Wno-deprecated -std=c++0x -Wall -pedantic -O3 -fopenmp -xHost
INC=-I$(EIGEN3_INCLUDE_DIR) -I./ -I/usr/local/cuda/include
LIB=-DGL_GLEXT_PROTOTYPES -lglut -lGL -lGLU
OBJECT = Box.o Image.o main.o Picture.o Pixel.o Ray.o Sphere.o

ifdef NOCUDA
	CFLAGS += -D NOCUDA
else
	CFLAGS += -lcuda -lcudart
endif

ifdef NOMP
	CFLAGS += -D NOMP
endif

ifdef NOPHI
	CFLAGS += -D NOPHI
endif

all: $(OBJECT)
	$(CC) -g $(CFLAGS) $(INC) $(OBJECT) $(LIB) -o rt

%.o: %.cpp
	$(CC) -g -c $< $(CFLAGS) $(INC) $(LIB)

%.cpp: %.h
	touch $@
	
%.cpp: %.hpp
	touch $@
	
%.o: %.cu
	$(CU) -g -c -m64 $< $(LIBS) $(OPTS)

%.cu: %.h
	touch $@

run:
	./rt
clean:
	rm -f *~ *.o a.out rt
clear:
	clear
	rm -f *~ *.o a.out rt
	$(CC) $(CFLAGS) $(INC) *.cpp $(LIB) -o rt
fast:
	rm -f *~ *.o a.out rt
	clear
	$(CC) $(CFLAGS) $(INC) *.cpp $(LIB) -o rt
	./rt
