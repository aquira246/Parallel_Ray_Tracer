CU=nvcc
CC=icpc
CFLAGS=-ansi -pedantic -Wno-deprecated -std=c++0x -Wall -pedantic -O3 -fopenmp -xHost -lcudadevrt -lcudart
LDFLAGS=-ansi -pedantic -Wno-deprecated -std=c++0x -Wall -pedantic -O3 -fopenmp -xHost -lcudadevrt -lcudart -o
INC=-I./ -I/usr/local/cuda/include
LIB=
CUFLAGS=-rdc=true

OBJECT = Image.o main.o Parse.o Picture.o Pixel.o Plane.o Ray.o Scene.o Sphere.o Shape.o Triangle.o Tokens.o VectorMath.o Vector3f.o

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
	nvcc -g -arch=sm_30 -ccbin=icpc -Xcompiler "$(LDFLAGS) $(INC) $(LIB)" $(OBJECT) -o rt

%.o: %.cpp
	$(CC) -g -c $< $(CFLAGS) $(INC) $(LIB)

%.cpp: %.h
	touch $@
	
%.cpp: %.hpp
	touch $@
	
%.o: %.cu
	$(CU) -g -c -m64 -arch=sm_30 $< $(LIBS) $(INC) $(CUFLAGS) $(OPTS)

%.cu: %.h
	touch $@

ball:
	./rt resources/bunny_small.pov
tri:
	./rt resources/bunny_small_tris.pov
triTest:
	./rt resources/triTest.pov
good:
	./rt resources/simp_cam.pov
good2:
	./rt resources/simp_cam2.pov
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
