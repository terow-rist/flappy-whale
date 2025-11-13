SOURCES = main.c shader.c obj_loader.c utils.c whale.c pipes.c
EXE = flappywhale_modern
CFLAGS = -Wall -g
LIBS = -lGL -lGLU -lglut -lGLEW -lm
LD = gcc
OBJECTS = $(SOURCES:.c=.o)

all: $(EXE)

$(EXE): $(OBJECTS)
	$(LD) $(OBJECTS) -o $(EXE) $(LIBS)

%.o: %.c
	$(LD) -c $(CFLAGS) $< -o $@

clean:
	rm -f $(EXE) $(OBJECTS)
