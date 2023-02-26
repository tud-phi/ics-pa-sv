import numpy as np
import cv2
import torch
import tonic
import os
from typing import Dict
from progressbar import progressbar
from matplotlib import animation
from typing import Tuple

from jax_double_pendulum.visualization import render_robot_cv2


def read_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    This method is for reading RGB image data from .npz file.
    Args:
        path: RGB images data path
    Return:
        state: ground truth, (num, len, 2)
        observation: RGB image data, (num, len, 3, 28, 28)
    """
    data = np.load(path)
    states = data["th_d_window_snn"]
    observation = data["th_pix_window_snn"]
    return states, observation


def get_diff(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    Get ON- and OFF- event-based image from the difference between two gray images
    Args:
        im1: RGB image at time step t, (size, size)
        im2: RGB image at time step t+1, (size, size)
    Return:
        img: two channel event-based image, (2, size, size)
    """
    xlength = im1.shape[0]
    ylength = im1.shape[1]
    img = np.full((2, xlength, ylength), 0, dtype=np.uint8)
    for i in range(xlength):
        for j in range(ylength):
            if int(im1[i][j]) - int(im2[i][j]) > 0:
                img[0][i][j] = 1
            if int(im2[i][j]) - int(im1[i][j]) > 0:
                img[1][i][j] = 1
    return img


def generate_snn_data(
    obs: np.ndarray, num_iter: int, num_data: int, time_window=20
) -> torch.FloatTensor:
    """
    Generate event-based data based on the input sequence of RGB images
    Args:
        obs: a sequence of RGB image, (num_iteration, 20*num_data+1, 3, 28, 28)
        num_iter: number of iterations
        num_data: number of event-based data should be generated
        time_window: the time window of the event-based data, default as 20
    Returns:
        snn_data: event_based data, (num_iteration, 20*num_data, 2, 28, 28)
    """
    snn_data = torch.tensor([])
    for e in range(num_iter):
        gray_img_ini = cv2.cvtColor(obs[e][0], cv2.COLOR_BGR2GRAY)
        snn_data_epoch = torch.tensor([])
        for t in range(1, int(time_window * num_data + 1)):
            gray_img = cv2.cvtColor(obs[e][int((t + 1) * 10 - 1)], cv2.COLOR_BGR2GRAY)
            img = torch.tensor(get_diff(gray_img, gray_img_ini)).unsqueeze(0)
            snn_data_epoch = torch.cat((snn_data_epoch, img))
            gray_img_ini = gray_img
        snn_data = torch.cat((snn_data, snn_data_epoch.unsqueeze(0)))
    return snn_data


def snn_animation(snn_data: torch.FloatTensor, path: str):
    """
    Save the animation of one event-based data
    Args:
        snn_data: input event-based data, (len, 2, size, size)
        path: save path
    """
    ani = tonic.utils.plot_animation(snn_data)
    pw_writer = animation.PillowWriter(fps=20)
    ani.save(path, writer=pw_writer)


class datasetOffline(torch.utils.data.Dataset):
    """
    generate a torch dataset for saving event-based data
    """

    def __init__(self, events, targets, num_iter, num_data, time_window):
        self.events = events
        self.targets = torch.tensor(targets)[:, 1:, :]
        self.num_iter = num_iter
        self.num_data = num_data
        self.time_window = time_window

    def __getitem__(self, index):
        i = int(index / self.num_data)
        j = int(index % self.num_data)
        events_output = torch.tensor([])
        for k in range(int(j * self.time_window), int((j + 1) * self.time_window)):
            events_output = torch.cat((events_output, self.events[i][k].unsqueeze(0)))
        target_output = self.targets[i][int((j + 1) * self.time_window * 10 - 1)]
        return events_output, target_output

    def __len__(self):
        return int(self.num_iter * self.num_data)


def divide_and_save_data(
    snn_data: torch.FloatTensor,
    state: np.ndarray,
    num_iter: int,
    num_data: int,
    path: str,
    time_window=20,
):
    """
    Divide the big event-based data into training data and save the event-based data with the ground truth locally
    Args:
        snn_data: input event-based data
        state: input ground truth
        num_iter: number of iterations
        num_data: number of event-based data that should be generated
        path: save path
        time_window: the time window of the event-based data, default as 20

    Returns:

    """
    dataset = datasetOffline(snn_data, state, num_iter, num_data, time_window)
    for i, (event, target) in enumerate(dataset):
        spike_save_path = "spike" + str(i) + ".pt"
        target_save_path = "target" + str(i) + ".pt"
        torch.save(event, os.path.join(path, spike_save_path))
        torch.save(target, os.path.join(path, target_save_path))


def draw_robot(
    _dataset: Dict[str, np.ndarray],
    rp: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """
    From values of the robots position elbow and end effector x-y position taken from the _dataset,
    return the _dataset with images of the robot at 'th_pix_ss' ".
    :param _dataset: dataset of simulated values.
    :param rp: Robot parameters dictionary
    """

    num_ic = _dataset["th_curr_ss"].shape[0]
    x_eb_ts = _dataset["x_eb_ts"]
    x_ts = _dataset["x_ts"]
    width = _dataset["th_pix_curr"].shape[-3]
    height = _dataset["th_pix_curr"].shape[-2]
    channel = _dataset["th_pix_curr"].shape[-1]

    imgs = np.zeros((num_ic, height, width, channel), dtype=np.uint8)

    for i in progressbar(range(num_ic)):
        imgs[i, :, :, :] = render_robot_cv2(
            rp, x_eb_ts[i, :], x_ts[i, :], height=height, width=width
        )

    _dataset["th_pix_curr"] = _dataset["th_pix_curr"].at[:].set(imgs)

    return _dataset


def draw_robot_snn(
    _dataset: Dict[str, np.ndarray],
    rp: Dict[str, float],
) -> Dict[str, np.ndarray]:
    """
    From simulated values of the robot over sequences,
    return the dataset with images of the robot at 'th_pix_window_snn' ".
    :param _dataset: dataset of simulated values.
    :param rp: Robot parameters dictionary
    """

    num_ic = _dataset["th_curr_ss"].shape[0]
    x_eb_ts = _dataset["x_eb_ts_window_snn"]
    x_ts = _dataset["x_ts_window_snn"]
    width = _dataset["th_pix_window_snn"].shape[-3]
    height = _dataset["th_pix_window_snn"].shape[-2]
    channel = _dataset["th_pix_window_snn"].shape[-1]
    window_size = _dataset["th_window_snn"].shape[1]

    imgs = np.zeros((num_ic, window_size, height, width, channel), dtype=np.float32)

    x_eb_ts = _dataset["x_eb_ts_window_snn"]
    x_ts = _dataset["x_ts_window_snn"]

    for i in progressbar(range(num_ic)):
        for j in range(window_size):
            imgs[i, j, :, :, :] = render_robot_cv2(
                rp, x_eb_ts[i, j, :], x_ts[i, j, :], height=height, width=width
            )

    _dataset["th_pix_window_snn"] = _dataset["th_pix_window_snn"].at[:].set(imgs)

    return _dataset
