{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc
    freeglut
    libGL
    libGLU
    mesa
  ];
}