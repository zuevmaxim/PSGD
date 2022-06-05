import sys

stepdecay_trials_length = 1
test_repeats = 10
nthreads = [2]
cluster_size = [1]

datasets = [
    # "a8a",
    # "covtype",
    # "webspam",
    # "music",
    "rcv1",
    # "epsilon",
    # "news20"
]
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


def generate_update_delays(nweights):
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


def create_step_decay_trials(d, n):
    stepdecay = get_step_decay(d)
    if n == 1:
        return [stepdecay]
    return [stepdecay ** ((i + 1) / stepdecay_trials_length) for i in range(0, stepdecay_trials_length * 2, 2)]


def get_step_decay(d):
    if d in stepdecay_per_dataset:
        return stepdecay_per_dataset[d]
    else:
        return stepdecay_per_dataset["default"]


dpi = 300

# def extract_time(f):
#     with open(f, "r") as file:
#         times = []
#         epochs = []
#         for line in file:
#             l = line.strip()
#             res = re.search(r'epoch: (\d+) train_time: ([\d.]+)', l)
#             if res:
#                 epochs.append(float(res.group(1)))
#                 times.append(float(res.group(2)))
#         times = times
#         epochs = epochs
#         # print(f, len(times))
#         return np.array(times), np.array(epochs)
#
#
# def extract_epoch_time(f):
#     with open(f, "r") as file:
#         times = []
#         epochs = []
#         for line in file:
#             l = line.strip()
#             res = re.search(r'epoch_time: ([\d.]+)', l)
#             if res:
#                 times.append(float(res.group(1)))
#         times = times
#         return np.array(times), np.array(epochs)
