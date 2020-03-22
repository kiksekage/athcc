from collections import defaultdict
from fuctools import reduce


from scipy.spatial.distance import euclidean
import pandas

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

