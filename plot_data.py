import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style='whitegrid')


def read_df(results_path):
    columns = ["algorithm", "threads", "cluster_size", "converge", "time", "train_acc", "validate_acc",
               "test_acc", "epochs", "epoch_time", "step_size", "step_decay", "update_delay", "target_acc",
               "block_size"]
    df = pd.read_csv(results_path, header=None, names=columns)

    if len(df[df["converge"] == 0]):
        print(f"{results_path} has unconverted results!!")

    df = df[df["converge"] == 1]

    return df


def plot_data(dataset, results_path):
    df = read_df(results_path)

    output_dir = os.path.dirname(results_path)

    # plot epochs
    sns.lineplot(x="threads", y="epochs", data=df, hue="algorithm", errorbar=('ci', 95))
    plt.title(f"{dataset} Iterations")
    plt.savefig(f"{output_dir}/{dataset}_iterations.png")
    plt.close()


if __name__ == "__main__":
    path = "results/svm_230312-224245"
    datasets = ["rcv1", "news20"]
    for d in datasets:
        plot_data(d, f"{path}/{d}.csv")
