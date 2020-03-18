import pandas
from pathlib import Path
import numpy as np
from scipy.spatial.distance import euclidean
from collections import defaultdict
from functools import reduce
import glob
import matplotlib.pyplot as plt

data_dir = Path("pointing-data-ATHCC")
collections = data_dir / "full_normalized" / "collections.csv"
speeds = Path("speeds")
collection_data = pandas.read_csv(collections)

points = [
    "indexFinger",
    "hand",
    "forearm",
    "upperArm",
    "rightShoulder",
    "hmd",
    "leftShoulder",
]


def deltatime(x2, x1):
    return x2.time - x1.time


def speed(x2, x1):
    speed_list = []
    time = deltatime(x2, x1)
    for point in points:
        point_x2_vals = x2[[f"{point}.X", f"{point}.Y", f"{point}.Z"]].values
        point_x1_vals = x1[[f"{point}.X", f"{point}.Y", f"{point}.Z"]].values
        point_speed = euclidean(point_x2_vals, point_x1_vals) / time
        speed_list.append((point, point_speed))
    speed_list.append(("cid", x2.cid))
    speed_list.append(("pid", x2.pid))
    speed_list.append(("time", x2.time))
    return speed_list

def new_speed(df, columns):
    df.index=pandas.to_timedelta(df["time"], unit='s')
    derivatives = df.diff().div(df.index.to_series().diff().dt.total_seconds(), axis=0)
    for c in derivatives.columns:
        if c not in columns:
            derivatives = derivatives.drop(c, axis=1)
    return derivatives

# dp = collection_data.query("pid==1 & cid==0")
"""
keep = ['indexFinger.X', 'indexFinger.Y', 'indexFinger.Z', 
       'hand.X', 'hand.Y', 'hand.Z', 'forearm.X', 'forearm.Y', 'forearm.Z', 
       'upperArm.X', 'upperArm.Y', 'upperArm.Z', 'rightShoulder.X', 
       'rightShoulder.Y', 'rightShoulder.Z', 'hmd.X', 'hmd.Y', 'hmd.Z', 
       'leftShoulder.X', 'leftShoulder.Y', 'leftShoulder.Z']
"""
def loader():
    for pid in range(1, int(collection_data.max(axis=0).pid) + 1):
        dfs = []
        for cid in range(0, int(collection_data.max(axis=0).cid) + 1):
            dp = collection_data.query(f"pid=={pid} & cid=={cid}")
            collected_dict = defaultdict(list)
            for i in range(1, dp.shape[0]):
                kv_list = speed(dp.iloc[i, :], dp.iloc[i - 1, :])
                for k, v in kv_list:
                    collected_dict[k].append(v)

            dfs.append(pandas.DataFrame.from_dict(collected_dict))
            print(cid)
        final = reduce(
            lambda left, right: pandas.concat([left, right], ignore_index=True), dfs
        )
        final.to_csv(speeds / f"pid{pid}.csv", index=True)

#dp.index=pandas.to_timedelta(dp["time"], unit='s')
#dp.diff().div(dp.index.to_series().diff().dt.total_seconds(), axis=0)

# loader()

df_dict = {}

#for idx, path in enumerate(sorted(speeds.glob("**/*.csv"))):
#    df_dict[idx] = pandas.read_csv(path)

def plotter(pid, cid, to_plot):
    data = collection_data.query(f"pid=={pid} & cid=={cid}")
    
    derivatives = new_speed(data, to_plot)

    rolled = derivatives.rolling('1s').mean()
    rolled = rolled.dropna()

    for column in rolled:
        plt.plot(rolled[column], label=column)

    plt.legend()
    plt.ylabel("Speed in m/s")
    plt.xlabel("Time in s")
    plt.show()

plotter(5,6,['indexFinger.X', 'indexFinger.Y', 'indexFinger.Z'])
#[max(x.indexFinger) for x in [df_dict[i] for i in range(len(df_dict))]]
