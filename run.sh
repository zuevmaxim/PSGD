#!/bin/bash

make clean
make

python3 collect_svm.py
#sudo shutdown now -h
