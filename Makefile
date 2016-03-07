CU=nvcc
CC=icpc
CFLAGS=-ansi -pedantic -Wno-deprecated -std=c++0x -Wall -pedantic -O3 -fopenmp -xHost
INC=-I$(EIGEN3_INCLUDE_DIR) -I./ -I/usr/local/cuda/include
LIB=-DGL_GLEXT_PROTOTYPES -lglut -lGL -lGLU

OBJECT = Image.o main.o Parse.o Picture.o Pixel.o Ray.o Scene.o Sphere.o Shape.o Triangle.o Tokens.o VectorMath.o

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

ifdef DEBUG
	CFLAGS += -D DEBUG
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
	./rt resources/bunny_small.pov
clean:
	rm -f *~ *.o a.out rt
clear: $(OBJECT)
	clear
	rm -f *~ *.o a.out rt
	$(CC) $(CFLAGS) $(INC) *.cpp $(LIB) -o rt
fast: $(OBJECT)
	rm -f *~ *.o a.out rt
	clear
	$(CC) $(CFLAGS) $(INC) *.cpp $(LIB) -o rt
	./rt resources/bunny_small.pov
