import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

sns.set_palette("bright")
sns.set(style='whitegrid')


def read__metrics_df(results_path):
    columns = ["algorithm", "threads", "cluster_count", "epoch", "accuracy"]
    df = pd.read_csv(results_path, header=None, names=columns)
    return df


def read_df(results_path):
    columns = ["algorithm", "threads", "cluster_count", "converge", "time", "train_acc", "validate_acc",
               "test_acc", "epochs", "epoch_time", "step_size", "step_decay", "update_delay", "target_acc",
               "block_size"]
    df = pd.read_csv(results_path, header=None, names=columns)

    if len(df[df["converge"] == 0]):
        print(f"{results_path} has unconverted results!!")

    df = df[df["converge"] == 1]

    return df


def plot_all(dataset, results_path):
    plot_data(dataset, results_path)
    plot__metrics_data(dataset, f"{results_path}.metric")


def plot_data(dataset, results_path):
    df = read_df(results_path)

    output_dir = os.path.dirname(results_path)

    # plot epochs
    sns.lineplot(x="threads", y="epochs", data=df, hue="algorithm", errorbar=('ci', 95))
    plt.title(f"{dataset} Iterations")
    plt.savefig(f"{output_dir}/{dataset}_iterations.png", dpi=300)
    plt.close()


def plot__metrics_data(dataset, results_path):
    df = read__metrics_df(results_path)

    output_dir = os.path.dirname(results_path)

    sns.lineplot(x="epoch", y="accuracy", data=df, hue=df[["algorithm", "threads", "cluster_count"]].apply(tuple, axis=1), errorbar=('ci', 95))
    plt.title(f"{dataset} Accuracy by epoch")
    plt.savefig(f"{output_dir}/{dataset}_accuracy.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    path = "results/svm_230319-145026"
    datasets = ["rcv1"]
    for d in datasets:
        # plot_data(d, f"{path}/{d}.csv")
        plot__metrics_data(d, f"{path}/{d}.csv.metric")
