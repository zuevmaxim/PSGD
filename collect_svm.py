#!/usr/bin/env python3

import subprocess
import sys
import time
from subprocess import check_call
from os import path

test_repeats = 10
phy_cores = 16
datasets = [
    # "a8a",
    # "covtype",
    # "webspam",
    # "music",
    "rcv1",
    # "epsilon",
    # "news20",
    # "url",
    # "kdda",
]
stepdecay_trials_length = 1
nthreads = [1, 2, 4, 8, 16]
algorithms = [
    "HogWild",
    "HogWild++",
    "MyWild"
]
max_iterations = {
    "HogWild": {"default": 150, "epsilon": 75, "kdda": 20},
    "HogWild++": {"default": 50, "epsilon": 25, "kdda": 10},
    "MyWild": {"default": 50, "epsilon": 25, "kdda": 10},
}
block_size = [2048]
maxstepsize = {
    "a8a": 5e-01,
    "covtype": 5e-03,
    "webspam": 2e-01,
    "music": 5e-08,
    "rcv1": 5e-01,
    "epsilon": 1e-01,
    "news20": 5e-01,
    "url": 5e-01,
    "kdda": 0.02,
}
target_accuracy = {
    "a8a": 0.845374,
    "covtype": 0.76291,
    "webspam": 0.92700,
    "rcv1": 0.978028,
    "epsilon": 0.89740,
    "news20": 0.96425,
    "url": 0.99,
    "kdda": 0.9424,
}
stepdecay_per_dataset = {
    "a8a": 0.8,
    "covtype": 0.85,
    "webspam": 0.8,
    "music": 0.8,
    "rcv1": 0.8,
    "epsilon": 0.85,
    "news20": 0.8,
    "url": 0.8,
    "kdda": 0.3,
    "default": 0.5,
}
use_permutation = [False, True]


def get_cluster_sizes(algorithm, threads):
    if algorithm == "HogWild":
        return [threads]
    return [threads // x for x in [2, 4, 8] if threads // x >= 1]


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
    # return [int(update_delay * (2 ** i)) for i in [-3, -1, 0, 1, 3]]


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


if __name__ == "__main__":
    output_dir = "results/svm_" + time.strftime("%m%d-%H%M%S")
    check_call("mkdir -p {}/".format(output_dir), shell=True)

    for d in datasets:
        output_file = "{}/{}.csv".format(output_dir, d)
        input_path = "{}/input_{}.txt".format(output_dir, d)
        input_file = open(input_path, 'w')
        for algorithm in algorithms:
            for thread in nthreads:
                phy_threads = min(thread, phy_cores)
                for cluster_size in get_cluster_sizes(algorithm, phy_threads):
                    clusters = phy_threads // cluster_size
                    if (phy_threads % cluster_size) != 0:
                        continue
                    epochs = get_epochs(d, max_iterations[algorithm])
                    epochs = get_effective_epochs(algorithm, clusters, epochs)
                    for update_delay in generate_update_delays(algorithm, clusters):
                        accuracy = target_accuracy[d]
                        step_size = maxstepsize[d]
                        for step_decay in create_step_decay_trials(d, algorithm, clusters):
                            for bs in block_size:
                                for permute in use_permutation:
                                    permutation = "permutations/{}/{}_{}.txt".format(d, clusters, phy_threads) if permute else "none"
                                    if permute and not path.exists(permutation):
                                        if len(use_permutation) > 1:
                                            continue
                                        permutation = "none"
                                    input_file.write("{} {} {} {} {} {} {} {} {} {} {}\n".format(
                                        algorithm, test_repeats, thread, cluster_size, epochs,
                                        update_delay, accuracy, step_size, step_decay, bs, permutation
                                    ))
        input_file.write("exit\n")
        input_file.close()
        verbose = "-v" in sys.argv
        v = " -v" if verbose else ""
        cmd_line = "bin/svm data/{} data/{}.t data/{}.t {} {}{}".format(d, d, d, output_file, input_path, v)
        if "-d" in sys.argv:
            cmd_line = "perf record --call-graph dwarf " + cmd_line
        print(cmd_line)
        if not is_dry_run():
            return_code = subprocess.Popen(cmd_line, shell=True).wait()
            if return_code != 0:
                print("Process failed: {}".format(return_code))
        else:
            print("*** This is a dry run. No results will be produced. ***")
        print()
