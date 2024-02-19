# Intelligent Control Systems (2024): Practical Assignment

The assignment can be completed either **locally** on your own machine (natively on macOS / Ubuntu or inside Docker container) or in the cloud leveraging **[GitHub Codespaces]((#2-github-codespaces))**. Accordingly, either follow the instructions in [Section 1](#1-native-installation), [Section 2](#2-github-codespaces), or [Section 3](#3-local-dev-containers).

## 1. Native installation (preferred)

The code should run natively on Ubuntu (preferred), macOS, and Windows. A native installation is the most resource efficient option and allows you to leverage the full power of your machine. However, it requires you to install all dependencies on your machine. Therefore, the setup is a bit more involved than with (some of) the other options.

### 1.1 Requirements

Our codebase relies heavily on [JAX](https://github.com/google/jax), which natively supports Linux `x86_64` (with NVIDIA GPU support), macOS hosts (both `x86_64` / Intel & `arm64` / Apple Silicon, no GPU support), and Windows (no GPU support on host, experimental support for NVIDIA GPUs on WSL2).

Furthermore, you can either use our provided scripts to install the required dependencies in a Conda environment (Option 1), or install all packages manually (Option 2).
We strongly recommend to use [Conda](https://docs.conda.io/en/latest/), as this will allow you to easily select the desired Python version and prevent any version conflicts.

#### 1.1.1 Important note for Windows users

JAX is now supported on Windows. Therefore, this codebase should also run on Windows. However, all the development and testing of this practical assignment was done on Ubuntu. Accordingly, we cannot guarantee that everything will work as expected on Windows and can provide only limited support for Windows users.
We recommend to setup a [dual boot installation with Windows & Ubuntu](https://linuxconfig.org/how-to-install-ubuntu-20-04-alongside-windows-10-dual-boot). Alternatively, also [Linux Subsystem for Windows](https://docs.microsoft.com/en-us/windows/wsl/about) can be used.

### 1.2 Clone the repository

Please download the code template from the [GitHub repository](https://github.com/tud-phi/ics-pa-sv) by clicking on _Code_ -> _Download ZIP_ or by cloning the repository using `git`:

```bash
git clone https://github.com/tud-phi/ics-pa-sv
```

### 1.3 Option 1: Installation using Conda (preferred)

We primarily support the installation of the required dependencies using Conda.
Please first install the latest version of Conda or Miniconda using the instructions on the
[Conda website](https://docs.conda.io/en/latest/miniconda.html).

Afterwards, please open a terminal and navigate to the top level directory of the repository using the `cd` command. Then, please follow the instructions below. **Important note for Windows users:** You might want to use the Anaconda Prompt instead of the Windows Command Prompt to make sure that the `conda` command is available.

#### 1.3.1 Create a new Conda environment and install dependencies

Then, run our bash script to create the new Conda environment `ics` with all required dependencies. On Ubuntu / macOS, you can do so by running:

```bash
./00-conda-setup.sh
```

If you encounter permission issues, please run `chmod +x /00-conda-setup.sh` to give executable permissions to the
bash script.

On Windows, please run (in the Anaconda Prompt):

```bash
.\00-conda-setup.bat
```

#### 1.3.2 Activate the Conda environment and add assignment folder to PYTHONPATH

Subsequently, we need to activate the Conda environment `ics` and add the assignment folder to the `PYTHONPATH` environment variable such that we can import Python modules from the assignment folder.

On Ubuntu / macOS, you can do so by running:

```bash
conda activate ics && ./02-add-to-pythonpath.sh
```

On Windows, please run (in the Anaconda Prompt):

```bash
conda activate ics && .\02-add-to-pythonpath.bat
```

#### 1.3.3 Install JAX with GPU support (optional)

If you are running on Ubuntu and want to leverage a NVIDIA GPU for increasing the LNN training speed in Problem 2c, you need to install JAX with GPU support.
**Please note that we will not offer any support for installing JAX with GPU support and it is totally up to your discretion to follow this installation step as the assignment can also be completed solely using the CPU.**

Please run in the `ics` Conda environment:

```bash
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

In case you are encountering issues with the installation or if JAX does not find your GPU, please refer to the [JAX README](https://github.com/google/jax#installation).

### 1.4 Option 2: Manual installation

We only support a manual installation of the required dependencies on Ubuntu. Please follow the instructions below to install all required dependencies manually.
This framework requires **Python 3.10/3.11**. For example, Ubuntu 22.04 ships with Python 3.10 and is therefore a suitable choice.
Please note that some required dependencies (for example PyTorch) might not be updated yet to work with Python 3.12 or newer.

#### 1.4.1 Install system packages

FFmpeg is required to create Matplotlib animations and save them as `.mp4` video files. Please follow installation instructions online such as [this one](https://www.hostinger.com/tutorials/how-to-install-ffmpeg). On Ubuntu, the package can be easily installed via:

```bash
sudo apt update && apt install -y ffmpeg
```

Node.js is required to install JupyterLab extensions for interactive plots. On Ubuntu, the package can be easily installed via:

```bash
sudo apt update && apt install -y nodejs
```

#### 1.4.2 Install Python dependencies

You can install the `jax_double_pendulum` package and all other necessary pip dependencies such as JAX, PyTorch, Matplotlib etc., by running the following command in the top level directory of the repository:

```bash
pip3 install .
```

If you encounter any issues with the installation of JAX, it is recommended to follow the installation instructions in the [JAX repository](https://github.com/google/jax#installation).

#### 1.4.3 Add assignment folder to PYTHONPATH

As we import some Python modules from folders outside a package, we need to add the assignment folder to the `PYTHONPATH` environment variable.

```bash
./02-add-to-pythonpath.sh
```

### 1.5 Usage

Please don't forget, if applicable, to activate the Conda environment before running any scripts or notebooks.

As mentioned in the assignment description, most of the code needs to be completed within the interactive Jupyter notebooks. You can use the following command to start a Jupyter notebook server in the top level directory of the repository:

```bash
jupyter notebook
```

## 2. GitHub Codespaces

A codespace is a development environment that's hosted in the cloud. Different instance types ranging from 2 CPU cores, 4 GB of RAM, and 32 GB of storage and up to 16 CPU cores, 32 GB of RAM and 128 GB of storage are available.

**Important:** 180 core-hours of GitHub Codespaces usage per month are included for free in _GitHub Pro_, which is offered as part of the _[GitHub Student Developer Pack](<https://education.github.com/pack>)_.
If you haven't already, please [register](https://education.github.com/benefits?type=student) for the pack using your TU Delft email address to get access to this free usage quota.
This will be sufficient for 90 hours per month of continuous usage with the smallest 2-core instance type.
**Make sure to stop your Codespaces instance when you are not using it to avoid unnecessary usage of your quota.**
Also, if you have used up the free quota, you will need to pay for any additional usage (not covered by this course).

**We strongly advise you to carefully study [Section 2.5](#25-commiting--pushing-with-git) to prevent any loss of code.**

### 2.1 Accessing the code for students

As you are studying this `README`, you probably know that the code template is available on GitHub in the
[`tud-phi/ics-pa-sv`](https://github.com/tud-phi/ics-pa-sv) repository.
Please click on _Use this template_ and then _Create new repository_ to create a new repository for the
assignment solution in your own personal GitHub account. **Please make sure to make the repository private.**

### 2.2 Open in GitHub Codespaces

Then, please open the new repository in a GitHub Codespaces instance by clicking on _Code_ -> _Open with Codespaces_.

### 2.3 Installation

No worries, all dependencies are automatically installed in the GitHub Codespaces environment. You should be able to start working right away.
The python executable is symlinked to `/usr/local/bin/python`.

**If you are encountering issues with missing Python dependencies or you observe a `jupyter: command not found` error when executing the `./10-start-notebook-as-student.sh` bash script, please run the following command in the GitHub Codespaces terminal:**

```bash
./01-pip-install.sh
```

Afterwards, the issues should be resolved.

### 2.4 Usage

You can open the Jupyter notebooks in the editor and then use the integrated Jupyter notebook extension to execute them. If you are prompted to select a kernel, please choose the kernel `Python 3.11.x /usr/local/bin/python`.

Alternatively, you can also start a Jupyter notebook server in the VS Code terminal, for which port-forwarding should be
configured automatically:

```bash
./10-start-notebook-as-student.sh
```

### 2.5 Commiting & pushing with git

Complimentary to saving your code on the Codespaces instance, you will also want to **push the code changes to your GitHub repository**, so that your code is not lost when the instance is deleted.
We refer to the internet for comprehensive guides on _git_. In the following, we will only point out the basic usage of pushing code from Codespaces to the _main_ branch of the GitHub repository:

1. Click on _Source Control_ in the left sidebar.
2. Here, you can see all files you have modified / added / deleted. You can _add_ / _stage_ changes by clicking on the `+` symbol right of the filename.
3. You can _commit_ the _Staged Changes_ by writing a concise message descriping the changes into the text box and then clicking on _Commit_.
4. Click on _Sync Changes_ or _Push_ to mirror the local commits to the remote GitHub repository.

### 2.6 Managing the memory usage

A standard 2-core instance of GitHub Codespaces has a memory limit of 4 GB.
This memory limit can be exhausted quite quickly, in particular when running neural network trainings. You can monitor the current memory usage of your instance by running `htop` in the terminal. When the instance runs out of memory, this usually results in a Jupyter notebook kernel crash. You would for example see the following or a very similar error message:

> The Kernel crashed while executing code in the the current cell or a previous cell. Please review the code in the cell(s) to identify a possible cause of the failure.

When you are running low on memory, please stop and close any Jupyter notebooks you might not be using at the moment. This will free some memory.

**If you keep encountering kernel crashes, we recommend upgrading to a larger GitHub Codespaces instance.** One option is the 4-core instance type with 8 GB of memory.
The various instance options are documented [here](https://docs.github.com/en/billing/managing-billing-for-github-codespaces/about-billing-for-github-codespaces#pricing-for-paid-usage). Please note that in the GitHub Student Developer Pack, you have [180 core-hours of Codespaces](https://docs.github.com/en/get-started/learning-about-github/githubs-products#github-pro) per month included for free. For a 4-core instance, this evaluates to 45 hours of continuous usage of the instance per month.
To change your instance to an 4-core instance, please click on _Code_ -> _Codespaces_ -> _On current branch_ -> _..._ -> _Change machine type_. Then, select the 4-core instance type.

### 2.7 Working with Matplotlib

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

## 3. Local Dev containers

Visual Studio Code Dev Containers allow you to open the repository in a containerized development environment.
It relies on [Docker](https://www.docker.com) to virtualize a pre-configured Ubuntu system with all necessary dependencies installed.
This allows you to execute code also on a **natively unsupported host operating system (e.g. Windows)** or to avoid installing all dependencies on your host system.
**Please note that we are currently not providing support for the Dev Container setup.**

In the following, we will describe the basic setup and usage of the Dev Container. You find a more comprehensive guide [here](https://code.visualstudio.com/docs/devcontainers/containers). Please also consult the [Docker documentation](https://docs.docker.com) for more information on Docker.

### 3.1 Install Docker

Please install [Docker Desktop](https://www.docker.com/products/docker-desktop) on Windows / Mac / [Ubuntu](https://docs.docker.com/desktop/install/ubuntu/). If you are one Windows, please make sure to install and enable the WSL 2 backend in the Docker Desktop settings.

### 3.2 Allocate sufficient resources to Docker

Docker runs in a Virtual Machine (VM) while limiting the available resources such as CPU cores, memory, and storage, which might result in insufficient resources for executing the code in this repository.
Accordingly, you might need to allocate more resources to the Docker VM than provided by default.

We recommend at least 4 CPUs, 6 GB of memory, and 32 GB of disk space. However, allocating too many resources to the VM might slow down / freeze your host system.

#### 3.2.1 Configure resource limits on Windows hosts with WSL 2 backend

If you have followed the provided indications and configured Docker to use WSL2, you will find out that you have no access to the advanced settings in Docker that allow you to limit the RAM and number of CPUs of your system (WSL does not provide a means to set the disk space). To change the resources of your containers when Docker is using WSL as backend, you need to make use of a `.wslconfig` file. To do so, you need to take the following steps (you require Visual Studio Code installed, see section 3.3 of this manual).

- In Visual Studio Code select in the top toolbar _File_ -> _New File..._ and name it `.wslconfig`. VS Code will prompt you to indicate the location to which you have to save the file. The file should be located under your user folder:

```
C:\Users\<your_user_name>\.wslconfig
```

- Once the file is open in VS Code, you need to make sure that the endline character of the file is _LF_. You can do that in the blue Status Bar at the bottom of the VS Code window. Towards the right there should be a piece of text indicating _CRLF_ or _LF_. If the text displayed is _LF_ then you don't need to perform any further action. If the text displayed is _CRLF_, then click on it and select _LF_ in the command palette at the top of the window.

- Copy and paste the following text into your `.wslconfig` file. Notice that if the computational resources of your computer are limited you may need to reduce some of these values (e.g. if you only have 4GB of RAM, you would need to reduce the _memory_ value, although that can result in issues while running the code). If you have a more powerful computer, feel free to increase the values without forgetting to leave some resources for your host system.

```
[wsl2]

memory=6GB

processors=4
```

- Save the changes (Ctrl + S) and close the window. If you have already installed Docker you will need to restart your computer for the changes to be effective.

- You can verify that the changes are taking effect by running the command below. The value displayed directly below _total_ should match the value you have assigned to `memory` in your `.wslconfig` file.

```bash
free --giga
```

#### 3.2.2 Configure resource limits on macOS / Ubuntu hosts

Please open Docker Desktop and go to _Settings_ -> _Resources_ -> _Advanced_ and modify the resources as described as in the Docker Desktop [macOS](https://docs.docker.com/desktop/settings/mac/#resources) and [Linux](https://docs.docker.com/desktop/settings/linux/#advanced) manuals.

### 3.3 Install Visual Studio Code

Please install [Visual Studio Code](https://code.visualstudio.com/) and subsequently also the [Dev Container extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers). With the Dev Containers extension installed, you will see a new green status bar item at the far bottom left.

### 3.4 Creating a private git repository on your own GitHub account

As you are studying this `README`, you probably know that the code template is available on GitHub in the
[`tud-phi/ics-pa-sv`](https://github.com/tud-phi/ics-pa-sv) repository.
Please click on _Use this template_ and then _Create new repository_ to create a new repository for the
assignment solution in your own personal GitHub account. **Please make sure to make the repository private.**

### 3.5 Clone your own repository

Please clone the repository you have just created on your own GitHub account to your local machine. For example, you can do so by opening a new window in VS Code, then clicking on _Source Control_ in the left sidebar, and then on _Clone Repository_.

### 3.6 Launch the Dev Container

Now, please open the folder with the cloned repository in Visual Studio Code. You will see a green status bar item at the far bottom left. Click on it and select _Reopen in Container_.
Alternatively, you might also be automatically prompted to reopen the folder in a container. This will use the settings in the `.devcontainer` folder to build a new Dev container and then open a new VS Code window with the repository opened in the containerized environment.

### 3.7 Install Python dependencies

After VS Code has been openend in the container, you will see a terminal window at the bottom of the screen. The process of installing Python dependencies should be started automatically. If not, please run the following command in the terminal:

```bash
./01-pip-install.sh
```

### 3.8 Usage

You can open the Jupyter notebooks in the editor and then use the integrated Jupyter notebook extension to execute them. If you are prompted to select a kernel, please choose the kernel `Python 3.11.x /usr/local/bin/python`.

Alternatively, you can also start a Jupyter notebook server in the VS Code terminal, for which port-forwarding should be
configured automatically:

```bash
./10-start-notebook-as-student.sh
```

### 3.9 Commiting & pushing with git

With respect to backing up your code, the same principle of GitHub Codespaces also applies to Dev Containers: Complimentary to saving your code on the Codespaces instance, you will also want to **push the code changes to your GitHub repository**, so that your code is not lost when the Dev container is (accidentally) deleted.
We refer to the internet for comprehensive guides on _git_. In the following, we will only point out the basic usage of pushing code from Codespaces to the _main_ branch of the GitHub repository:

1. Click on _Source Control_ in the left sidebar.
2. Here, you can see all files you have modified / added / deleted. You can _add_ / _stage_ changes by clicking on the `+` symbol right of the filename.
3. You can _commit_ the _Staged Changes_ by writing a concise message descriping the changes into the text box and then clicking on _Commit_.
4. Click on _Sync Changes_ or _Push_ to mirror the local commits to the remote GitHub repository.

## 4. Jupyter notebook - tips & tricks

### 4.1 Reloading functions implemented in another notebook

When changing the content of functions implemented in a Jupyter notebook and used in other notebooks, it might (sometimes) be necessary to save all notebooks and then restarting the notebook kernel(s). This procedure will allow the function in all notebooks relying on it to be re-loaded.

### 4.2 Validating your implementation

Zou are able to validate the syntax of your code, the removal of all `NotImplementedError` exceptions, and the
passing of all public tests.

On Ubuntu / macOS, you can do so by running:

```bash
./20-validate-assignment.sh
```

On Windows, please run (in the Anaconda Prompt):

```bash
.\20-validate-assignment.bat
```

## 5. Preparing your submission

### 5.1 Validating your solution

First, make sure to double check the following:

1. Your contact information is filled in the corresponding cells at the top of the notebooks. Please do so as well for partly completed and incomplete notebooks.
2. Make sure that you have NOT INSERTED NOR DELETED any cells in any of the notebooks, to avoid potential issues with the autograding.
3. Make sure that your notebooks run top-to-bottom (or as far as you have implemented) after restarting the kernel. The provided script `20-validate-assignment.sh` / `20-validate-assignment.bat` will do this for you (see [Section 4.2](#42-validating-your-implementation)).

### 5.2 Creating a zip archive

Please create a zip archive of the entire repository folder. The name of the submitted ZIP file should be in the format studentNumber_FirstName_LastName.zip to facilitate the identification of the owner of the code. Please, stick to that convention by:

- Omitting middle names
- Omitting accents (e.g.  Rubén --> Ruben)
- Merging compound first/last names using CamelCase style (e.g 1234567_Ruben_MartinRodriguez.zip or 1112233_Name_vanDenSurname.zip)

In order to make sure that we receive all necessary metadata and that the structure of the archive is as expected, please use the provided script for this:

```bash
python ./25-create-submission-archive.py
```

Afterwards, you will find the zip archive on the top level of the repository folder. Please upload this zip archive to Brightspace.
