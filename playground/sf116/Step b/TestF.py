# Test with Time Series forecast
# see: https://keras.io/examples/timeseries/timeseries_traffic_forecasting/

import os
import typing

import numpy as np
import pandas as pd
import tensorflow
from keras import layers
from keras.preprocessing.timeseries import timeseries_dataset_from_array
from matplotlib import pyplot as plt
from tensorflow import keras


def load_test_data():
    url = "https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/data_loader/PeMS-M.zip"
    data_dir = keras.utils.get_file(origin=url, extract=True, archive_format="zip")
    data_dir = data_dir.rstrip(".zip")

    route_distances = pd.read_csv(os.path.join(data_dir, "W_228.csv"), header=None).to_numpy()
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
    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train: (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val):] - mean) / std

    return train_array, val_array, test_array


train_array, val_array, test_array = preprocess(speeds_array, train_size, val_size)

print(f"train set size: {train_array.shape}")
print(f"validation set size: {val_array.shape}")
print(f"test set size: {test_array.shape}")

batch_size = 64
input_sequence_length = 12
forecast_horizon = 3
multi_horizon = False


def create_tf_dataset(data_array: np.ndarray, input_sequence_length: int, forecast_horizon: int, batch_size: int = 128,
                      shuffle=True, multi_horizon=True, ):
    inputs = timeseries_dataset_from_array(np.expand_dims(data_array[:-forecast_horizon], axis=-1), None,
                                           sequence_length=input_sequence_length, shuffle=False,
                                           batch_size=batch_size, )

    target_offset = (input_sequence_length if multi_horizon else input_sequence_length + forecast_horizon - 1)
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(data_array[target_offset:], None, sequence_length=target_seq_length,
                                            shuffle=False, batch_size=batch_size, )

    dataset = tensorflow.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()


train_dataset, val_dataset = (
    create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size)
    for data_array in [train_array, val_array]
)

test_dataset = create_tf_dataset(
    test_array,
    input_sequence_length,
    forecast_horizon,
    batch_size=test_array.shape[0],
    shuffle=False,
    multi_horizon=multi_horizon,
)


def compute_adjacency_matrix(route_distances: np.ndarray, sigma2: float, epsilon: float):
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 10000.0
    w2, w_mask = (route_distances * route_distances, np.ones([num_routes, num_routes]) - np.identity(num_routes),)
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask


def compute_adjacency_matrix(
        route_distances: np.ndarray, sigma2: float, epsilon: float
):
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 10000.0
    w2, w_mask = (
        route_distances * route_distances,
        np.ones([num_routes, num_routes]) - np.identity(num_routes),
    )
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask


class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes


class GraphConv(layers.Layer):
    def __init__(
            self,
            in_feat,
            out_feat,
            graph_info: GraphInfo,
            aggregation_type="mean",
            combination_type="concat",
            activation: typing.Optional[str] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = tensorflow.Variable(
            initial_value=keras.initializers.glorot_uniform()(
                shape=(in_feat, out_feat), dtype="float32"
            ),
            trainable=True,
        )
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbour_representations: tensorflow.Tensor):
        aggregation_func = {
            "sum": tensorflow.math.unsorted_segment_sum,
            "mean": tensorflow.math.unsorted_segment_mean,
            "max": tensorflow.math.unsorted_segment_max,
        }.get(self.aggregation_type)

        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )

        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features: tensorflow.Tensor):
        return tensorflow.matmul(features, self.weight)

    def compute_aggregated_messages(self, features: tensorflow.Tensor):
        neighbour_representations = tensorflow.gather(features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbour_representations)
        return tensorflow.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: tensorflow.Tensor, aggregated_messages: tensorflow.Tensor):
        if self.combination_type == "concat":
            h = tensorflow.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        return self.activation(h)

    def call(self, features: tensorflow.Tensor):
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)


sigma2 = 0.1
epsilon = 0.5
adjacency_matrix = compute_adjacency_matrix(route_distances, sigma2, epsilon)
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix.shape[0],
)
print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")


class LSTMGC(layers.Layer):
    """Layer comprising a convolution layer followed by LSTM and dense layers."""

    def __init__(
            self,
            in_feat,
            out_feat,
            lstm_units: int,
            input_seq_len: int,
            output_seq_len: int,
            graph_info: GraphInfo,
            graph_conv_params: typing.Optional[dict] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)

        # graph conv layer
        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None,
            }
        self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)

        self.lstm = layers.LSTM(lstm_units, activation="relu")
        self.dense = layers.Dense(output_seq_len)

        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

    def call(self, inputs):
        inputs = tensorflow.transpose(inputs, [2, 0, 1, 3])

        gcn_out = self.graph_conv(inputs)
        shape = tensorflow.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_feat = (shape[0], shape[1], shape[2], shape[3],)

        gcn_out = tensorflow.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))
        lstm_out = self.lstm(gcn_out)

        dense_output = self.dense(lstm_out)
        output = tensorflow.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
        return tensorflow.transpose(output, [1, 2, 0])


in_feat = 1
batch_size = 64
epochs = 20
input_sequence_length = 12
forecast_horizon = 3
multi_horizon = False
out_feat = 10
lstm_units = 64
graph_conv_params = {"aggregation_type": "mean", "combination_type": "concat", "activation": None, }

st_gcn = LSTMGC(in_feat, out_feat, lstm_units, input_sequence_length, forecast_horizon, graph, graph_conv_params, )
inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))
outputs = st_gcn(inputs)

model = keras.models.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0002), loss=keras.losses.MeanSquaredError(), )
model.fit(train_dataset, validation_data=val_dataset, epochs=epochs,
          callbacks=[keras.callbacks.EarlyStopping(patience=10)], )

x_test, y = next(test_dataset.as_numpy_iterator())
y_pred = model.predict(x_test)
plt.figure(figsize=(18, 6))
plt.plot(y[:, 0, 0])
plt.plot(y_pred[:, 0, 0])
plt.legend(["actual", "forecast"])
plt.show()
print(model.summary())

naive_mse, model_mse = (
    np.square(x_test[:, -1, :, 0] - y[:, 0, :]).mean(),
    np.square(y_pred[:, 0, :] - y[:, 0, :]).mean(),
)
print(f"naive MAE: {naive_mse}, model MAE: {model_mse}")
