#!/usr/bin/env python3

import subprocess
import sys
import time
from subprocess import check_call

from common import *

output_dir = "results/svm_" + time.strftime("%m%d-%H%M%S")
check_call("mkdir -p {}/".format(output_dir), shell=True)

stepdecay_trials_length = 1
nthreads = [1, 2, 4, 8, 16, 32, 64, 128]
algorithms = [
    "HogWild",
    "HogWild++",
]
max_iterations = {
    "HogWild": {"default": 150, "epsilon": 75},
    "HogWild++": {"default": 50, "epsilon": 25},
}
maxstepsize = {
    "a8a": 5e-01,
    "covtype": 5e-03,
    "webspam": 2e-01,
    "music": 5e-08,
    "rcv1": 5e-01,
    "epsilon": 1e-01,
    "news20": 5e-01,
}
target_accuracy = {
    "a8a": 0.845374,
    "covtype": 0.76291,
    "webspam": 0.92700,
    "rcv1": 0.97713,
    "epsilon": 0.89740,
    "news20": 0.96425,
}
stepdecay_per_dataset = {
    "a8a": 0.8,
    "covtype": 0.85,
    "webspam": 0.8,
    "music": 0.8,
    "rcv1": 0.8,
    "epsilon": 0.85,
    "news20": 0.8,
    "default": 0.5,
}


def get_clusters(algorithm, threads):
    if algorithm == "HogWild":
        return [threads]
    return [y for y in [threads // x for x in [2, 4]] if y >= 1]


def generate_update_delays(algorithm, nweights):
    if algorithm == "HogWild":
        return [0]
    if nweights <= 4:
        update_delay = 64
    elif nweights <= 10:
        update_delay = 16
    else:
        update_delay = 4
    return [update_delay]
    # return [update_delay * (2 ** i) for i in [-3, -1, 0, 1, 3]]


def get_epochs(d, iterations):
    if d in iterations:
        return iterations[d]
    else:
        return iterations["default"]


def is_dry_run():
    dryrun = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "-n":
            dryrun = True
        if sys.argv[1] == "-y":
            dryrun = False
    return dryrun


def create_step_decay_trials(d, algorithm, c):
    stepdecay = get_step_decay(d)
    if algorithm == "HogWild":
        return [stepdecay]
    return [stepdecay ** (1 / c)]
    # return [stepdecay ** ((i + 1) / stepdecay_trials_length) for i in range(0, stepdecay_trials_length * 2, 2)]


def get_step_decay(d):
    if d in stepdecay_per_dataset:
        return stepdecay_per_dataset[d]
    else:
        return stepdecay_per_dataset["default"]


def get_effective_epochs(a, c, e):
    if a == "HogWild":
        return e
    effective_epochs = e * c
    effective_epochs = min(1000, effective_epochs)
    effective_epochs = max(150, effective_epochs)
    return effective_epochs


for d in datasets:
    output_file = "{}/{}.csv".format(output_dir, d)
    input_path = "{}/input_{}.txt".format(output_dir, d)
    input_file = open(input_path, 'w')
    for algorithm in algorithms:
        for thread in nthreads:
            phy_threads = min(thread, phy_cores)
            clusters = get_clusters(algorithm, phy_threads)
            for cluster in clusters:
                cluster_count = phy_threads // cluster
                if (phy_threads % cluster) != 0:
                    continue
                epochs = get_epochs(d, max_iterations[algorithm])
                epochs = get_effective_epochs(algorithm, cluster_count, epochs)
                for update_delay in generate_update_delays(algorithm, cluster_count):
                    accuracy = target_accuracy[d]
                    step_size = maxstepsize[d]
                    for step_decay in create_step_decay_trials(d, algorithm, cluster_count):
                        input_file.write("{} {} {} {} {} {} {} {} {}\n"
                                         .format(algorithm, test_repeats, thread, cluster, epochs,
                                                 update_delay, accuracy, step_size, step_decay))
    input_file.write("exit\n")
    input_file.close()
    cmd_line = "bin/svm data/{} data/{}.t data/{}.t {} {}".format(d, d, d, output_file, input_path)
    print(cmd_line)
    if not is_dry_run():
        subprocess.Popen(cmd_line, shell=True).wait()
    else:
        print("*** This is a dry run. No results will be produced. ***")
    print()
