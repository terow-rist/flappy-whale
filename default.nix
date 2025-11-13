{ pkgs ? import <nixpkgs> {} }: pkgs.mkShell {
  buildInputs = with pkgs; [ gcc freeglut glew libGL libGLU mesa ];
}
