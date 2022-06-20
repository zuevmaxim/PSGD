#!/usr/bin/env python3
import numpy as np

from collect_svm import *

if __name__ == "__main__":
    output_dir = "results/best_svm_" + time.strftime("%m%d-%H%M%S")
    check_call("mkdir -p {}/".format(output_dir), shell=True)

    for d in datasets:
        output_file = "{}/{}.csv".format(output_dir, d)
        input_path = "{}/input_{}.txt".format(output_dir, d)
        input_file = open(input_path, 'w')
        epochs = 30
        target_score = 0.9424
        for step_size in np.arange(0.01, 0.04, 0.01):
            for step_decay in np.arange(0.1, 1, 0.2):
                input_file.write("{} {} {} {} {} {} {} {} {} {}\n".format(
                    "HogWild", 3, 1, 1, epochs, 1, target_score, step_size, step_decay, 1
                ))
        input_file.write("exit\n")
        input_file.close()
        cmd_line = "bin/svm data/{} data/{}.t data/{}.t {} {}".format(d, d, d, output_file, input_path)
        print(cmd_line)
        if not is_dry_run():
            subprocess.Popen(cmd_line, shell=True).wait()
        else:
            print("*** This is a dry run. No results will be produced. ***")
        print()
