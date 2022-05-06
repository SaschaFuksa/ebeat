# Test with Time Series forecast
# see: https://keras.io/examples/timeseries/timeseries_traffic_forecasting/

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras


def load_test_data():
    url = "https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/data_loader/PeMS-M.zip"
    data_dir = keras.utils.get_file(origin=url, extract=True, archive_format="zip")
    data_dir = data_dir.rstrip(".zip")

    route_distances = pd.read_csv(
        os.path.join(data_dir, "W_228.csv"), header=None
    ).to_numpy()
    speeds_array = pd.read_csv(os.path.join(data_dir, "V_228.csv"), header=None).to_numpy()

    print(f"route_distances shape={route_distances.shape}")
    print(f"speeds_array shape={speeds_array.shape}")

    return route_distances, speeds_array


route_distances, speeds_array = load_test_data()

sample_routes = [
    0, 1, 4, 7, 8, 11, 15, 108, 109, 114, 115, 118, 120, 123, 124, 126, 127, 129, 130, 132, 133, 136, 139, 144, 147,
    216,
]
route_distances = route_distances[np.ix_(sample_routes, sample_routes)]
speeds_array = speeds_array[:, sample_routes]

print(f"route_distances shape={route_distances.shape}")
print(f"speeds_array shape={speeds_array.shape}")

plt.figure(figsize=(18, 6))
plt.plot(speeds_array[:, [0, -1]])
plt.legend(["route_0", "route_25"])
plt.show()

plt.figure(figsize=(8, 8))
plt.matshow(np.corrcoef(speeds_array.T), 0)
plt.xlabel("road number")
plt.ylabel("road number")
plt.show()

train_size, val_size = 0.5, 0.2


def preprocess(data_array: np.ndarray, train_size: float, val_size: float):
    """Splits data into train/val/test sets and normalizes the data.

    Args:
        data_array: ndarray of shape `(num_time_steps, num_routes)`
        train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the train split.
        val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the validation split.

    Returns:
        `train_array`, `val_array`, `test_array`
    """

    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train : (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val) :] - mean) / std

    return train_array, val_array, test_array


train_array, val_array, test_array = preprocess(speeds_array, train_size, val_size)

print(f"train set size: {train_array.shape}")
print(f"validation set size: {val_array.shape}")
print(f"test set size: {test_array.shape}")
