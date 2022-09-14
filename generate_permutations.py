#!/usr/bin/env python3

from collect_svm import *
import subprocess
from os import path

if __name__ == "__main__":
    for d in datasets:
        output_dir = "permutations/{}".format(d)
        check_call("mkdir -p {}/".format(output_dir), shell=True)
        for parts in nthreads:
            if parts == 1:
                continue
            output_file = "{}/{}.txt".format(output_dir, parts)
            if path.exists(output_file):
                print("Permutation for dataset {} in {} parts is already generated, remove {} file to rerun"
                      .format(d, parts, output_file))
                continue

            cmd_line = "bin/analysis {} data/{} {} -v".format(parts, d, output_file)
            print(cmd_line)
            subprocess.Popen(cmd_line, shell=True).wait()
