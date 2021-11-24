#!/bin/bash
# Script that build module using g++ compiler
# Path to Python.h directory: /usr/include/python3.7m
# Path to static/dynamic library: "/usr/lib"
g++ -fPIC -I "/usr/include/python3.7m" -I "/home/avkozlovskiy_1/.local/lib/python3.6/site-packages/numpy/core/include/numpy" -std=c++17 -c src/pymodule/main.cpp -o main.o
g++ -shared -I "/usr/lib" -o pymodule.so -o test/pymodule.so main.o
rm main.o