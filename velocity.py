from pathlib import Path


import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

model_path = Path(".")
data_dir = Path("pointing-data-ATHCC")
collections = data_dir / "full_normalized" / "collections.csv"
targets = data_dir / "full_normalized" / "targets.csv"
speeds = Path("speeds")

collection_data = pandas.read_csv(collections)
labels = pandas.read_csv(targets)


def derivative(df, columns):
    df.index = pandas.to_timedelta(df["time"], unit="s")
    derivatives = df.diff().div(df.index.to_series().diff().dt.total_seconds(), axis=0)
    for c in derivatives.columns:
        if c not in columns:
            derivatives = derivatives.drop(c, axis=1)
    derivatives = derivatives.dropna()
    return derivatives


def generate_dataset(keep, smooth=False):
    data = []
    for pid in range(1, int(collection_data.max(axis=0).pid) + 1):
        print(f"Loading data for participant with id {pid}")
        for cid in range(0, int(collection_data.max(axis=0).cid) + 1):
            df = collection_data.query(f"pid=={pid} & cid=={cid}")
            label = (
                labels.query(f"pid=={pid} & cid=={cid}")
                .drop(["cid", "pid"], axis=1)
                .values[0]
            )

            derivs = derivative(df, keep)
            if smooth:
                derivs = derivs.rolling("200ms").mean()
            data.append((derivs, label))
    return data


def plotter(pid, cid, to_plot, smooth=True):
    data = collection_data.query(f"pid=={pid} & cid=={cid}")

    derivatives = derivative(data, to_plot)

    if smooth:
        rolled = derivatives.rolling("200ms").mean()
        derivatives = rolled.dropna()

    for column in derivatives:
        plt.plot(derivatives[column], label=column)

    plt.legend()
    plt.ylabel("Velocity in m/s")
    plt.xlabel("Time in s")
    plt.show()


"""
plotter(
    1,
    50,
    ["indexFinger.X", "indexFinger.Y", "indexFinger.Z", "hand.X", "hand.Y", "hand.Z"],
    False,
)
"""
