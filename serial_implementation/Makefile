CU=nvcc
CC=icpc
CFLAGS=-ansi -pedantic -Wno-deprecated -std=c++0x -Wall -pedantic -O3 -fopenmp -xHost
INC=-I$(EIGEN3_INCLUDE_DIR) -I./ -I/usr/local/cuda/include
LIB=-DGL_GLEXT_PROTOTYPES -lglut -lGL -lGLU

OBJECT = Image.o main.o Parse.o Picture.o Pixel.o Plane.o Ray.o Scene.o Sphere.o Shape.o Triangle.o Tokens.o VectorMath.o

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

ball:
	./rt ../resources/bunny_small.pov
tri:
	./rt ../resources/bunny_small_tris.pov
good:
	./rt ../resources/simp_cam.pov
good2:
	./rt ../resources/simp_cam2.pov
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
	./rt ../resources/bunny_small.pov
