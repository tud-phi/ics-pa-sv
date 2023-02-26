# Intelligent Control Systems: Practical Assignment

The assignment can be completed either **locally** on your own machine (macOS or Ubuntu) or in the cloud using **GitHub
Codespaces**. Accordingly, either follow the instructions in [Section 1](#1-local-system) or in [Section 2](#2-github-codespaces).

## 1. Local system

### 1.1. Requirements

As our codebase relies on [JAX](https://github.com/google/jax), we only support Linux `x86_64` or macOS hosts (both `x86_64` / Intel & `arm64` / Apple Silicon).
**We therefore strongly recommend to use either one of the two supported host environment, or complete the assignment on [GitHub Codespaces](#2-github-codespaces) instead.**

Furthermore, you can either use our provided scripts to install the required dependencies in a Conda environment (Option 1), or install all packages manually (Option 2).
We strongly recommend to use [Conda](https://docs.conda.io/en/latest/), as this will allow you to easily select the desired Python version and prevent any version conflicts.

#### 1.1.1. Important note for Windows users

Windows is not officially supported by JAX, which we depend on in this codebase.
We quote from the [JAX README](https://github.com/google/jax):
> Windows users can use JAX on CPU and GPU via the Windows Subsystem for Linux.
> In addition, there is some initial community-driven native Windows support, but since it is still somewhat immature,
> there are no official binary releases and it must be [built from source for Windows](https://jax.readthedocs.io/en/latest/developer.html#additional-notes-for-building-jaxlib-from-source-on-windows).
> For an unofficial discussion of native Windows builds, see also the [Issue #5795 thread](https://github.com/google/jax/issues/5795).

While it is possible to install [Linux Subsystem for Windows](https://docs.microsoft.com/en-us/windows/wsl/about)
or alternatively configure a [dual boot setup with Windows & Ubuntu](https://linuxconfig.org/how-to-install-ubuntu-20-04-alongside-windows-10-dual-boot),
we are not able to offer any assistance with this. Therefore, we strongly recommend to use one of the other supported environments.

### 1.2.Option 1: Installation using Conda

We primarily support the installation of the required dependencies using Conda.
Please first install the latest version of Conda or Miniconda using the instructions on the
[Conda website](https://docs.conda.io/en/latest/miniconda.html).

#### 1.2.1 Create a new Conda environment and install dependencies

Then, run our bash script to create the new Conda environment `ics` with all required dependencies:

```bash
./00-conda-setup.sh
```

If you encounter permission issues, please run `chmod +x /00-conda-setup.sh` to give executable permissions to the
bash script.

#### 1.2.2 Activate the Conda environment and add assignment folder to PYTHONPATH

Subsequently, you can activate the Conda environment `ics` by running:

```bash
conda activate ics && ./02-add-to-pythonpath.sh
```

#### 1.2.3 Install PyTorch with GPU support (optional)

If you want to leverage a NVIDIA GPU for increasing the neural network training speed in Problem 1, you need to install PyTorch with GPU support.
**Please note that we will not offer any support for installing PyTorch with GPU support and it is totally up to your discretion to follow this installation step as the assignment can also be completed solely using the CPU.**

As documented in the _[PyTorch Get Started](https://pytorch.org/get-started/locally/)_ guide, please run in the `ics` Conda environment:

```bash
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
```

#### 1.2.4 Install JAX with GPU support (optional)

If you want to leverage a NVIDIA GPU for increasing the LNN training speed in Problem 2c, you need to install JAX with GPU support.
**Please note that we will not offer any support for installing JAX with GPU support and it is totally up to your discretion to follow this installation step as the assignment can also be completed solely using the CPU.**

Please run in the `ics` Conda environment:

```bash
conda install jax cuda-nvcc -c conda-forge -c nvidia
```

In case you are encountering issues with the installation or if JAX does not find your GPU, please refer to the [JAX README](https://github.com/google/jax#installation).


### 1.3. Option 2: Manual installation

This framework requires **Python 3.10**. Please note that some required dependencies might not be updated yet to work with Python 3.11.

#### 1.3.1. Install ffmpeg

FFmpeg is required to create Matplotlib animations and save them as `.mp4` video files. Please follow installation instructions online such as [this one](https://www.hostinger.com/tutorials/how-to-install-ffmpeg). On Ubuntu, the package can be easily installed via:

```bash
sudo apt update && apt install -y ffmpeg 
```

#### 1.3.2. Install Python dependencies

You can install the `jax_double_pendulum` package and all necessary dependencies by running the following command in the top level directory of the repository:

```bash
pip install .
```

#### 1.3.3. Add assignment folder to PYTHONPATH

As we import some Python modules from folders outside a package, we need to add the assignment folder to the `PYTHONPATH` environment variable.

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

If you encounter any issues with the installation of JAX, it is recommended to follow the installation instructions in the [JAX repository](https://github.com/google/jax#installation).

### 1.4. Usage for students

Please don't forget, if applicable, to activate the Conda environment before running any scripts.

Generally, all Python scripts can be executed from the top level directory of the repository. For example:

```bash
python examples/main.py
```

For Jupyter notebooks, you can start use our script to start a Jupyter notebook server in the terminal:

```bash
./10-start-notebook-as-student.sh
```

## 2. GitHub Codespaces

**Important:** The usage GitHub Codespaces is included for free in the _GitHub Student Developer Pack_ (<https://education.github.com/pack>).
If you haven't already, please [register](https://education.github.com/benefits?type=student) for the pack using your TU Delft email address to get access to GitHub Codespaces.

### 2.1 Accessing the code for students

As you are studying this `README`, you probably know that the code template is available on GitHub in the
[`tud-cor-sr/ics-pa-template`](https://github.com/tud-cor-sr/ics-pa-sv) repository.
Please click on _Use this template_ and then _Create new repository_ to create a new repository for the
assignment solution in your own personal GitHub account. **Please make sure to make the repository private.**

### 2.2 Open in GitHub Codespaces

Then, please open the new repository in GitHub Codespaces by clicking on _Code_ -> _Open with Codespaces_.

### 2.3. Installation

No worries, all dependencies are already installed in the GitHub Codespaces environment. You can start working right away.
All Python dependencies are available for the default Python interpreter, which is located at `/usr/local/bin/python`.

### 2.4. Usage for students

If you want to run Python scripts in the GitHub Codespaces environment, you can use the integrated VS Code terminal.
For Jupyter notebooks, you can open the notebook in the editor and then use the integrated Jupyter notebook extension.
Alternatively, you can also start a Jupyter notebook server in the VS Code terminal, for which port-forwarding should be
configured automatically:

```bash
./10-start-notebook-as-student.sh
```

### 2.6. Working with Matplotlib

There are two ways to work with Matplotlib plots within standard Python scripts (i.e. not Jupyter notebooks):

1. Add the following line to the top of your Python script, which allows you to run the script interactively in VS Code.

```python
# %%
# Now you can add the rest of your code here
import matplotlib.pyplot as plt
plt.figure()
plt.show()
```

Then you can run the script interactively by clicking on the _Run Cell_ button in the top left corner of the code cell.

2. Instead of showing the Matplotlib code, you can also just save the plot to a file and open it in a new tab in VS Code.

```python
import matplotlib.pyplot as plt
plt.figure()
plt.savefig("my_plot.png")
```

## 3. Jupyter notebook - tips & tricks

### 3.1 Reloading functions implemented in another notebook

When changing the content of functions implemented in a Jupyter notebook and used in other notebooks, it might (sometimes) be necessary to save all notebooks and then restarting the notebook kernel(s). This procedure will allow the function in all notebooks relying on it to be re-loaded.

### 3.2 Validating your implementation

You are able to validate the syntax of your code, the removal of all `NotImplementedError` exceptions, and the
passing of all public tests by running the following command in the top level directory of the repository:

```bash
AUTOGRADING=true nbgrader validate assignment/**/**.ipynb
```
