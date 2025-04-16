from matplotlib import pyplot as plt
from jax import numpy as jnp
import numpy as np
import torch
from typing import Optional


def process_data(df):
    # load the saved data
    df = df.iloc[1:-1]

    # produce new df with only the necesary data by dropping the unncecessary ones
    df = df.drop(
        columns=[
            "tau_ff_ts_1",
            "tau_ff_ts_2",
            "x_d_ts_1",
            "x_d_ts_2",
            "x_dd_ts_1",
            "x_dd_ts_2",
            "x_eb_ts_1",
            "x_eb_ts_2",
            "x_ts_1",
            "x_ts_2",
        ]
    )
    # keep tau fb 'tau_fb_ts_1','tau_fb_ts_2'

    # evaluate delta_th and delta_th_d (reference_th - th and referencunwrappede_th_d - th_d) for task 3
    # NOTE this must now not be unwrapped
    # df['delta_th_d_1'] =   df['ref_th_d_1'] - df['th_d_ts_1']
    # df['delta_th_d_2'] =   df['ref_th_d_2'] - df['th_d_ts_2']
    # df['delta_th_1'] =     df['ref_th_1'] - df['th_ts_1']
    # df['delta_th_2'] =     df['ref_th_2'] - df['th_ts_2']

    # wrap the angles around -pi,+pi

    df["th_ts_2_rel"] = np.angle(np.exp(1j * (df["th_ts_2"] - df["th_ts_1"])))
    df["th_d_ts_2_rel"] = df["th_d_ts_2"] - df["th_d_ts_1"]
    df["th_ts_2"] = np.angle(np.exp(1j * df["th_ts_2"]))
    df["th_ts_1"] = np.angle(np.exp(1j * df["th_ts_1"]))
    # if "ref_th_2" in df.columns and "ref_th_1" in df.columns:
    #     df["th_ts_2_rel"] = np.angle(np.exp(1j * (df["ref_th_2"] - df["th_ts_1"])))
    #     df["th_d_ts_2_rel"] = df["ref_th_2"] - df["th_ts_1"]
    # # now wrap the delta theta angles if necessary
    # df['delta_th_1'] = np.angle(np.exp(1j * df['delta_th_1']))
    # df['delta_th_2'] = np.angle(np.exp(1j * df['delta_th_2']))

    return df


def plot_data(df, input_columns, output_columns, filepath: Optional[str] = None):
    data_to_plot = input_columns + output_columns
    rows = int(len(data_to_plot) / 2)

    fig, axes = plt.subplots(rows, 2, figsize=(10, 6), constrained_layout=True)
    plt.suptitle("Training inputs and labels")

    plots = [
        (title, column, i // 2, i % 2)
        for i, (title, column) in enumerate(zip(data_to_plot, data_to_plot))
    ]

    for title, column, row, col in plots:
        ax = axes[row, col]
        ax.set_title(title)
        ax.plot(df["t_ts"], df[column], label=title, color="lightblue")
        ax.legend()

    if filepath is not None:
        plt.savefig(filepath)


def generate_training_data(df, input_cols, output_cols):
    # Training on deltas
    train_x = torch.tensor(df[input_cols].to_numpy(), dtype=torch.float32)

    # Produce y labels only feedback action
    train_y = torch.tensor(df[output_cols].to_numpy(), dtype=torch.float32)

    # Make into column vectors
    train_y = train_y.unsqueeze(1) if len(output_cols) == 1 else train_y

    # Make contiguous
    train_x = train_x.contiguous()
    train_y = train_y.contiguous()

    return train_x, train_y


def split_2d_columns(dict):
    # Create a new dictionary for the split columns
    new_data = {}

    # Iterate through the original dictionary
    for key, value in dict.items():
        if isinstance(value, jnp.ndarray) and value.ndim == 2:
            # Create new keys for the individual columns
            new_keys = [f"{key}_{i+1}" for i in range(value.shape[1])]
            # Update the new dictionary with the individual columns
            new_data.update(zip(new_keys, value.T))
        else:
            # If not a 2D array, simply copy the key-value pair
            new_data[key] = value

    return new_data
