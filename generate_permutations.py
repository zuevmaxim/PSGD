#!/usr/bin/env python3

from collect_svm import *
import subprocess
from os import path

if __name__ == "__main__":
    for algorithm in algorithms:
        for d in datasets:
            output_dir = "permutations/{}".format(d)
            check_call("mkdir -p {}/".format(output_dir), shell=True)
            for threads in nthreads:
                phy_threads = min(threads, phy_cores)
                for cluster_size in get_cluster_sizes(algorithm, phy_threads):
                    clusters = phy_threads // cluster_size
                    if (phy_threads % cluster_size) != 0:
                        continue
                    parts = phy_threads // clusters
                    if parts == 1:
                        continue
                    output_file = "{}/{}_{}.txt".format(output_dir, clusters, phy_threads)
                    if path.exists(output_file):
                        print("Permutation for dataset {} in {} clusters and {} parts is already generated, remove {} file to rerun"
                              .format(d, clusters, parts, output_file))
                        continue

                    cmd_line = "bin/analysis {} {} data/{} {}".format(clusters, parts, d, output_file)
                    print(cmd_line)
                    return_code = subprocess.Popen(cmd_line, shell=True).wait()
                    if return_code != 0:
                        print("Process failed: {}".format(return_code))
