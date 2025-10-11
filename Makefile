SOURCES = main.c whale.c pipes.c utils.c
EXE = flappywhale

CFLAGS = -Wall
LIBS = -lGL -lGLU -lglut -lm

LD = gcc

OBJECTS = $(SOURCES:%.c=%.o)

default: all

all: $(EXE)

$(EXE): $(OBJECTS)
	$(LD) $(OBJECTS) -o $(EXE) $(LIBS)

clean:
	-rm -f $(EXE)
	-rm -f $(OBJECTS)
