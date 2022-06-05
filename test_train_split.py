#!/usr/bin/env pypy

import os
import random
import sys


def save(lines, out):
    for line in lines:
        print(line, end="", file=out)


if len(sys.argv) < 5:
    os.exit(1)

fraction = float(sys.argv[1])
input_file = sys.argv[2]
output_file_1 = sys.argv[3]
output_file_2 = sys.argv[4]

with open(input_file, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)
    index = int(len(lines) * fraction)
    with open(output_file_1, 'w') as out:
        save(lines[:index], out)
    with open(output_file_2, 'w') as out:
        save(lines[index:], out)
